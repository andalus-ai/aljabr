package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

import java.util.Optional;

/**
 * One decoder forward pass: {@code token, position -> logits}, given a
 * running {@link KVCache}.
 *
 * <p>Supports standard LLaMA/Mistral/Qwen models as well as Gemma family models
 * (Gemma 2, Gemma 3, Gemma 4) with embedding scaling, per-head Q/K/V RMSNorm,
 * post-attention and post-FFN RMSNorm, GeGLU activations, per-layer SWA / RoPE,
 * rope_freqs proportional rotary embedding, per-layer token embeddings, layer
 * output scaling, and final logit softcapping.</p>
 *
 * <p>Gemma 4 SWA layers use <em>bidirectional</em> attention during prefill
 * (all tokens in the prompt attend to each other within the sliding window),
 * matching the {@code LLAMA_NON_CAUSAL_TYPE_SWA_ONLY} behaviour in llama.cpp.
 * Use {@link #prefill(int[], KVCache)} for prompt processing and
 * {@link #forward(int, int, KVCache, boolean)} for single-token decode.</p>
 */
final class GgufDecodeStep {
    private final GGUFModel model;
    private final GgufModelConfig cfg;

    private final GGUFTensorInfo tokenEmbed;
    private final float[] outputNorm;
    private final GGUFTensorInfo outputWeight;

    // Per-layer token embeddings (Gemma 4)
    private final GGUFTensorInfo perLayerTokEmbd;
    private final GGUFTensorInfo perLayerModelProj;
    private final float[] perLayerProjNorm;
    private final GGUFTensorInfo[] perLayerInpGate;
    private final GGUFTensorInfo[] perLayerProj;
    private final float[][] perLayerPostNorm;

    private final float[][] attnNorm;
    private final float[][] attnQNorm;
    private final float[][] attnKNorm;
    private final float[][] attnVNorm;
    private final float[][] postAttnNorm;
    private final float[][] ropeFreqs;
    private final float[] outScale;
    private final GGUFTensorInfo[] wQ, wK, wV, wO;

    private final float[][] ffnNorm;
    private final float[][] postFfnNorm;
    private final GGUFTensorInfo[] ffnGate, ffnUp, ffnDown;

    // Scratch buffers sized to worst-case (maximum) across layers
    private final float[] x, xb, q, k, v, attnFull, attnOut, gate, up, hb, scores, logits, ffnOutBuf;
    private final float[][] headScores;
    private final float[] perLayerRawTok, perLayerModelProjOut, perLayerInputs;
    private final float[] peGate, peOut, pePostNormBuf;

    // Max KV heads across all layers (real max, not clamped to nH)
    private final int maxKVHeads;

    // Full-GPU single command buffer decode step available
    private final boolean canUseMetalDecodeStep;

    GgufDecodeStep(GGUFModel model, GgufModelConfig cfg) {
        this.model = model;
        this.cfg = cfg;

        int dim = cfg.embeddingDim();
        int ffnDim = cfg.ffnDim();
        int nH = cfg.nHeads();
        int hDim = cfg.headDim();
        int layers = cfg.nLayers();
        int nEmbdPerLayer = cfg.nEmbdPerLayer();

        // Compute true max KV heads across all layers (ignore nKVHeads which may be wrong)
        int maxKVH = 1;
        if (cfg.nKVHeadsPerLayer() != null) {
            for (int kvh : cfg.nKVHeadsPerLayer()) maxKVH = Math.max(maxKVH, kvh);
        } else {
            maxKVH = cfg.nKVHeads();
        }
        this.maxKVHeads = maxKVH;

        this.tokenEmbed = requireTensor("token_embd.weight");
        this.outputNorm = dequantizeVector(requireTensor("output_norm.weight"), dim);
        this.outputWeight = findTensor("output.weight").orElse(tokenEmbed); // tied embeddings

        // Per-layer token embedding tensors
        this.perLayerTokEmbd = findTensor("per_layer_token_embd.weight").orElse(null);
        this.perLayerModelProj = findTensor("per_layer_model_proj.weight").orElse(null);
        this.perLayerProjNorm = findTensor("per_layer_proj_norm.weight")
                .map(t -> dequantizeVector(t, nEmbdPerLayer > 0 ? nEmbdPerLayer : 1))
                .orElse(null);

        this.perLayerInpGate = new GGUFTensorInfo[layers];
        this.perLayerProj = new GGUFTensorInfo[layers];
        this.perLayerPostNorm = new float[layers][];

        this.attnNorm = new float[layers][];
        this.attnQNorm = new float[layers][];
        this.attnKNorm = new float[layers][];
        this.attnVNorm = new float[layers][];
        this.postAttnNorm = new float[layers][];
        this.ropeFreqs = new float[layers][];
        this.outScale = new float[layers];
        this.wQ = new GGUFTensorInfo[layers];
        this.wK = new GGUFTensorInfo[layers];
        this.wV = new GGUFTensorInfo[layers];
        this.wO = new GGUFTensorInfo[layers];

        this.ffnNorm = new float[layers][];
        this.postFfnNorm = new float[layers][];
        this.ffnGate = new GGUFTensorInfo[layers];
        this.ffnUp = new GGUFTensorInfo[layers];
        this.ffnDown = new GGUFTensorInfo[layers];

        for (int i = 0; i < layers; i++) {
            String p = "blk." + i + ".";
            int layerHDim = cfg.headDimForLayer(i);
            attnNorm[i] = dequantizeVector(requireTensor(p + "attn_norm.weight"), dim);
            attnQNorm[i] = findTensor(p + "attn_q_norm.weight").map(t -> dequantizeVector(t, layerHDim)).orElse(null);
            attnKNorm[i] = findTensor(p + "attn_k_norm.weight").map(t -> dequantizeVector(t, layerHDim)).orElse(null);
            attnVNorm[i] = findTensor(p + "attn_v_norm.weight").map(t -> dequantizeVector(t, layerHDim)).orElse(null);
            postAttnNorm[i] = findTensor(p + "post_attention_norm.weight").map(t -> dequantizeVector(t, dim)).orElse(null);

            // Proportional RoPE factors: blk.N.rope_freqs.weight or global rope_freqs.weight
            // Only apply when tensor dimension matches the rotary half-dimension (layerHDim / 2)
            GGUFTensorInfo layerRope = findTensor(p + "rope_freqs.weight").orElse(null);
            if (layerRope != null) {
                ropeFreqs[i] = dequantizeVector(layerRope, layerHDim / 2);
            } else {
                ropeFreqs[i] = findTensor("rope_freqs.weight")
                        .filter(t -> t.shape().length > 0 && t.shape()[0] == layerHDim / 2)
                        .map(t -> dequantizeVector(t, layerHDim / 2))
                        .orElse(null);
            }

            // Optional layer output scale
            outScale[i] = findTensor(p + "layer_output_scale.weight")
                    .map(t -> {
                        float[] s = dequantizeVector(t, 1);
                        return s.length > 0 ? s[0] : 1.0f;
                    })
                    .orElse(1.0f);

            wQ[i] = requireTensor(p + "attn_q.weight");
            wK[i] = findTensor(p + "attn_k.weight").orElse(null);
            wV[i] = findTensor(p + "attn_v.weight").orElse(null);
            wO[i] = requireTensor(p + "attn_output.weight");

            ffnNorm[i] = dequantizeVector(requireTensor(p + "ffn_norm.weight"), dim);
            postFfnNorm[i] = findTensor(p + "post_ffw_norm.weight").map(t -> dequantizeVector(t, dim)).orElse(null);
            ffnGate[i] = findTensor(p + "ffn_gate.weight").orElse(null);
            ffnUp[i] = requireTensor(p + "ffn_up.weight");
            ffnDown[i] = requireTensor(p + "ffn_down.weight");

            // Per-layer embedding blocks
            if (perLayerTokEmbd != null) {
                perLayerInpGate[i] = findTensor(p + "inp_gate.weight").orElse(null);
                perLayerProj[i] = findTensor(p + "proj.weight").orElse(null);
                perLayerPostNorm[i] = findTensor(p + "post_norm.weight")
                        .map(t -> dequantizeVector(t, dim))
                        .orElse(null);
            }
        }

        // Register static layer norms on GPU bridge once to enable zero-copy decode slices
        if (GgufMetalBridge.isAvailable()) {
            GgufMetalBridge.registerLayerNorms(layers, dim, hDim, postAttnNorm, ffnNorm, postFfnNorm, attnNorm, attnQNorm, attnKNorm);
        }

        this.x = new float[dim];
        this.xb = new float[dim];
        this.q = new float[nH * hDim];
        this.k = new float[maxKVHeads * hDim];
        this.v = new float[maxKVHeads * hDim];
        this.attnFull = new float[nH * hDim];
        this.attnOut = new float[hDim];
        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];
        this.hb = new float[ffnDim];
        this.scores = new float[cfg.contextLength()];
        this.headScores = new float[nH][cfg.contextLength()];
        this.logits = new float[cfg.vocabSize()];
        this.ffnOutBuf = new float[dim];

        int peSize = nEmbdPerLayer > 0 ? nEmbdPerLayer * layers : 1;
        this.perLayerRawTok = new float[peSize];
        this.perLayerModelProjOut = new float[peSize];
        this.perLayerInputs = new float[peSize];
        this.peGate = new float[nEmbdPerLayer > 0 ? nEmbdPerLayer : 1];
        this.peOut = new float[dim];
        this.pePostNormBuf = new float[dim];

        if (GgufMetalBridge.isAvailable()) {
            GgufMetalBridge.ensureModelRegistered(model);
        }

        // Initialize full-GPU decode step pipeline if supported
        boolean eligible = !Boolean.getBoolean("aljabr.metal.decode.disable")
                && GgufMetalBridge.isDecodeStepAvailable()
                && (perLayerTokEmbd == null || nEmbdPerLayer <= 0);
        for (int l = 0; l < layers && eligible; l++) {
            if (wQ[l] == null || wO[l] == null || ffnUp[l] == null || ffnDown[l] == null) {
                eligible = false;
            }
        }

        boolean metalDecodeActive = false;
        if (eligible) {
            int[] strides = new int[layers];
            for (int i = 0; i < layers; i++) {
                int layerNKVH = cfg.nKVHeadsForLayer(i);
                int hD = cfg.headDimForLayer(i);
                strides[i] = layerNKVH * hD;
            }
            if (GgufMetalBridge.initKVCache(layers, cfg.contextLength(), strides)) {
                boolean allLayersRegistered = true;
                for (int l = 0; l < layers && allLayersRegistered; l++) {
                    int layerNKVH = cfg.nKVHeadsForLayer(l);
                    int layerHDim = cfg.headDimForLayer(l);
                    float layerRopeFreq = cfg.ropeFreqBaseForLayer(l);
                    boolean isSwa = cfg.isSwaLayer(l);
                    int n0 = nH * layerHDim;
                    int n1 = layerNKVH * layerHDim;
                    int n2 = layerNKVH * layerHDim;
                    int wO_K = n0;
                    int wO_N = dim;
                    int ffnIntermediate = ffnDim;
                    int stride = strides[l];

                    float attnScale = cfg.attentionScale();
                    if (attnScale <= 0f) {
                        attnScale = (attnQNorm[l] != null || attnKNorm[l] != null) ? 1.0f : (1.0f / (float) Math.sqrt(layerHDim));
                    }

                    boolean reg = GgufMetalBridge.registerDecodeLayer(
                            l, model,
                            wQ[l], wK[l], wV[l], wO[l],
                            ffnGate[l], ffnUp[l], ffnDown[l],
                            n0, n1, n2, wO_K, wO_N, ffnIntermediate,
                            nH, layerNKVH, layerHDim, stride,
                            layerRopeFreq, isSwa ? 1 : 0,
                            outScale[l], attnScale,
                            postAttnNorm[l] != null ? 1 : 0,
                            postFfnNorm[l] != null ? 1 : 0,
                            attnQNorm[l] != null ? 1 : 0,
                            attnKNorm[l] != null ? 1 : 0,
                            (cfg.hasUnitVNorm() || attnVNorm[l] != null) ? 1 : 0);
                    if (reg && ropeFreqs[l] != null) {
                        GgufMetalBridge.registerRopeFreqs(l, ropeFreqs[l]);
                    }
                    if (!reg) allLayersRegistered = false;
                }
                if (allLayersRegistered) {
                    metalDecodeActive = GgufMetalBridge.registerOutput(
                            model, outputNorm, outputWeight, dim, cfg.vocabSize(), cfg.finalLogitSoftcap());
                }
            }
        }
        this.canUseMetalDecodeStep = metalDecodeActive;
        if (metalDecodeActive) {
            System.out.println("[GgufDecodeStep] Full-GPU single command buffer decode step ENABLED (" + layers + " layers).");
        }
    }

    /**
     * Batch-prefill all prompt tokens through the network in one shot.
     *
     * <p>For Gemma 4 SWA layers this uses <em>bidirectional</em> attention
     * within the sliding window (matching llama.cpp's
     * {@code LLAMA_NON_CAUSAL_TYPE_SWA_ONLY}).  Global (dense) layers remain
     * strictly causal.  All KV pairs are deposited into {@code cache} so that
     * subsequent single-token decode calls can build on the prefill.
     *
     * @param tokenIds prompt token IDs, length T
     * @param cache    KV cache to populate (must be empty / positioned at 0)
     * @return logits array for the last token
     */
    float[] prefill(int[] tokenIds, KVCache cache) {
        int T   = tokenIds.length;
        int dim = cfg.embeddingDim();
        int nH  = cfg.nHeads();
        int ffnDim   = cfg.ffnDim();
        int layers   = cfg.nLayers();
        int nEmbdPL  = cfg.nEmbdPerLayer();
        float eps    = cfg.rmsNormEps();
        float embScale = cfg.embeddingScale();

        // ── 1. Token embeddings: X[T][dim] ───────────────────────────────────
        float[][] X = new float[T][dim];
        for (int t = 0; t < T; t++) {
            GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenIds[t], X[t]);
            if (embScale > 0f && embScale != 1.0f) {
                GGUFVectorOps.scale(X[t], embScale, dim);
            }
        }

        // ── 2. Per-layer token embedding projection (Gemma 4 n_embd_per_layer) ─
        // Currently n_embd_per_layer_input = 0 in the 12B GGUF, so this block
        // is skipped.  Left in as a placeholder for future models.
        boolean hasPerLayer = perLayerTokEmbd != null && nEmbdPL > 0;
        float[][] perLayerInputsT = null;   // [T][nEmbdPL * layers]
        if (hasPerLayer) {
            perLayerInputsT = new float[T][nEmbdPL * layers];
            float tokEmbdScale = (float) Math.sqrt(nEmbdPL);
            float projScale    = 1.0f / (float) Math.sqrt(dim);
            float inputScale   = 1.0f / (float) Math.sqrt(2.0);
            float[] rawTok     = new float[nEmbdPL * layers];
            float[] projOut    = new float[nEmbdPL * layers];
            for (int t = 0; t < T; t++) {
                GgufTensorOps.dequantizeRow(model, perLayerTokEmbd, tokenIds[t], rawTok);
                GGUFVectorOps.scale(rawTok, tokEmbdScale, nEmbdPL * layers);
                dispatchMatVec(perLayerModelProj, X[t], projOut);
                GGUFVectorOps.scale(projOut, projScale, nEmbdPL * layers);
                if (perLayerProjNorm != null) {
                    for (int l = 0; l < layers; l++) {
                        int off = l * nEmbdPL;
                        GGUFVectorOps.rmsNorm(projOut, off, perLayerProjNorm, projOut, off, nEmbdPL, eps);
                    }
                }
                for (int j = 0; j < nEmbdPL * layers; j++) {
                    perLayerInputsT[t][j] = (projOut[j] + rawTok[j]) * inputScale;
                }
            }
        }

        // ── 3. Layer loop ─────────────────────────────────────────────────────
        // Scratch: Q[T][nH*hDim], K[T][nKVH*hDim], V[T][nKVH*hDim]
        int maxHDim   = cfg.headDim();
        float[][] Qall = new float[T][nH * maxHDim];
        float[][] Kall = new float[T][maxKVHeads * maxHDim];
        float[][] Vall = new float[T][maxKVHeads * maxHDim];

        // Reusable batch buffers across layers (0 allocations inside 48-layer loop)
        float[] allXb    = new float[T * dim];
        float[] allQ     = new float[T * nH * maxHDim];
        float[] allK     = new float[T * maxKVHeads * maxHDim];
        float[] allV     = new float[T * maxKVHeads * maxHDim];
        float[] allAttn  = new float[T * nH * maxHDim];
        float[] allWoOut = new float[T * dim];
        float[] allFfnIn = new float[T * dim];
        float[] allFfnOut = new float[T * dim];
        float[] xbL      = new float[dim];
        float[] qLocal   = new float[nH * maxHDim];
        float[] kLocal   = new float[maxKVHeads * maxHDim];
        float[] vLocal   = new float[maxKVHeads * maxHDim];
        float[] gateL    = new float[ffnDim];
        float[] upL      = new float[ffnDim];
        float[] hbL      = new float[ffnDim];
        float[] attnScores = new float[T];
        float[] attnOutLocal = new float[maxHDim];
        float[][] Attn   = new float[T][nH * maxHDim];

        for (int layer = 0; layer < layers; layer++) {
            int layerNKVH     = cfg.nKVHeadsForLayer(layer);
            int layerHDim     = cfg.headDimForLayer(layer);
            float layerRopeFreq = cfg.ropeFreqBaseForLayer(layer);
            float[] layerRopeFreqs = ropeFreqs[layer];
            boolean isSwa     = cfg.isSwaLayer(layer);

            float scale;
            if (cfg.attentionScale() > 0f) {
                scale = cfg.attentionScale();
            } else if (attnQNorm[layer] != null || attnKNorm[layer] != null) {
                scale = 1.0f;
            } else {
                scale = (float) (1.0 / Math.sqrt(layerHDim));
            }

            int qDim = nH * layerHDim;
            int kvDim = layerNKVH * layerHDim;

            // ── 3a. QKV projections for all T tokens ─────────────────────────
            for (int t = 0; t < T; t++) {
                GGUFVectorOps.rmsNorm(X[t], 0, attnNorm[layer], allXb, t * dim, dim, eps);
            }

            long tQkv0 = System.nanoTime();
            if (wK[layer] != null && wV[layer] != null) {
                dispatchMatVecTripleBatch(wQ[layer], allQ, wK[layer], allK, wV[layer], allV, allXb, T);
            } else {
                for (int t = 0; t < T; t++) {
                    System.arraycopy(allXb, t * dim, xbL, 0, dim);
                    if (wK[layer] != null) {
                        dispatchMatVecPair(wQ[layer], qLocal, wK[layer], kLocal, xbL);
                        System.arraycopy(kLocal, 0, vLocal, 0, kvDim);
                    } else {
                        dispatchMatVec(wQ[layer], xbL, qLocal);
                    }
                    System.arraycopy(qLocal, 0, allQ, t * qDim, qDim);
                    System.arraycopy(kLocal, 0, allK, t * kvDim, kvDim);
                    System.arraycopy(vLocal, 0, allV, t * kvDim, kvDim);
                }
            }
            long tQkv = System.nanoTime() - tQkv0;

            for (int t = 0; t < T; t++) {
                System.arraycopy(allQ, t * qDim, qLocal, 0, qDim);
                System.arraycopy(allK, t * kvDim, kLocal, 0, kvDim);
                System.arraycopy(allV, t * kvDim, vLocal, 0, kvDim);

                if (attnQNorm[layer] != null) {
                    int qNormLen = attnQNorm[layer].length;
                    for (int h = 0; h < nH; h++) {
                        GGUFVectorOps.rmsNorm(qLocal, h * layerHDim, attnQNorm[layer], qLocal, h * layerHDim, qNormLen, eps);
                    }
                }
                GGUFVectorOps.rope(qLocal, nH, layerHDim, t, layerRopeFreq, layerRopeFreqs);
                System.arraycopy(qLocal, 0, Qall[t], 0, qDim);
                System.arraycopy(qLocal, 0, allQ, t * qDim, qDim);

                if (wK[layer] != null) {
                    if (attnKNorm[layer] != null) {
                        int kNormLen = attnKNorm[layer].length;
                        for (int kvH = 0; kvH < layerNKVH; kvH++) {
                            GGUFVectorOps.rmsNorm(kLocal, kvH * layerHDim, attnKNorm[layer], kLocal, kvH * layerHDim, kNormLen, eps);
                        }
                    }
                    if (attnVNorm[layer] != null) {
                        int vNormLen = attnVNorm[layer].length;
                        for (int kvH = 0; kvH < layerNKVH; kvH++) {
                            GGUFVectorOps.rmsNorm(vLocal, kvH * layerHDim, attnVNorm[layer], vLocal, kvH * layerHDim, vNormLen, eps);
                        }
                    } else if (cfg.hasUnitVNorm()) {
                        for (int kvH = 0; kvH < layerNKVH; kvH++) {
                            GGUFVectorOps.rmsNormUnit(vLocal, kvH * layerHDim, vLocal, kvH * layerHDim, layerHDim, eps);
                        }
                    }

                    GGUFVectorOps.rope(kLocal, layerNKVH, layerHDim, t, layerRopeFreq, layerRopeFreqs);
                    System.arraycopy(kLocal, 0, Kall[t], 0, kvDim);
                    System.arraycopy(kLocal, 0, allK, t * kvDim, kvDim);
                    System.arraycopy(vLocal, 0, Vall[t], 0, kvDim);
                    System.arraycopy(vLocal, 0, allV, t * kvDim, kvDim);

                    // Store into KV cache at position t
                    cache.storeDirect(layer, t, kLocal, vLocal, kvDim);
                }
            }

            // ── 3b. Attention for all T tokens ───────────────────────────────
            long tAttn0 = System.nanoTime();
            int swWin  = cfg.slidingWindow();
            boolean gpuAttnDone = false;
            // Use Metal FlashAttention-4 (MPSGraph SDPA) when prompt fits within sliding window (or non-SWA)
            if ((!isSwa || swWin <= 0 || T <= swWin) && GgufMetalBridge.isAvailable()) {
                gpuAttnDone = GgufMetalBridge.fa4Attention(allAttn, allQ, allK, allV, 1, T, T, nH, layerNKVH, layerHDim, scale, true);
            }

            if (!gpuAttnDone) {
                for (int t = 0; t < T; t++) {
                    int tStart;
                    int tEnd;
                    if (isSwa && swWin > 0) {
                        tStart = Math.max(0, t - swWin + 1);
                        tEnd   = t;
                    } else {
                        tStart = 0;
                        tEnd   = t;
                    }
                    int activeLen = tEnd - tStart + 1;

                    for (int h = 0; h < nH; h++) {
                        int kvH  = h / (nH / layerNKVH);
                        int qOff = h * layerHDim;
                        int kvOff = kvH * layerHDim;

                        // Compute attention scores
                        for (int j = tStart; j <= tEnd; j++) {
                            float dot = 0f;
                            for (int d = 0; d < layerHDim; d++) {
                                dot += Qall[t][qOff + d] * Kall[j][kvOff + d];
                            }
                            attnScores[j - tStart] = dot * scale;
                        }
                        GGUFVectorOps.softmax(attnScores, activeLen);

                        // Weighted sum of V
                        java.util.Arrays.fill(attnOutLocal, 0, layerHDim, 0f);
                        for (int j = tStart; j <= tEnd; j++) {
                            float w = attnScores[j - tStart];
                            for (int d = 0; d < layerHDim; d++) {
                                attnOutLocal[d] += w * Vall[j][kvOff + d];
                            }
                        }
                        System.arraycopy(attnOutLocal, 0, Attn[t], h * layerHDim, layerHDim);
                    }
                }
            }
            long tAttn = System.nanoTime() - tAttn0;

            // ── 3c. Output projection + residual + FFN for all T tokens ──────
            long tWo0 = System.nanoTime();
            if (!gpuAttnDone) {
                for (int t = 0; t < T; t++) {
                    System.arraycopy(Attn[t], 0, allAttn, t * qDim, qDim);
                }
            }
            dispatchMatVecBatch(wO[layer], allAttn, allWoOut, T);

            for (int t = 0; t < T; t++) {
                System.arraycopy(allWoOut, t * dim, xbL, 0, dim);
                if (postAttnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(xbL, postAttnNorm[layer], xbL, dim, eps);
                }
                GGUFVectorOps.addInPlace(X[t], xbL, dim);
            }
            long tWo = System.nanoTime() - tWo0;

            // FFN
            long tFfn0 = System.nanoTime();
            for (int t = 0; t < T; t++) {
                GGUFVectorOps.rmsNorm(X[t], 0, ffnNorm[layer], allFfnIn, t * dim, dim, eps);
            }

            boolean fusedSuccess = false;
            if (ffnGate[layer] != null) {
                fusedSuccess = dispatchFfnFusedBatch(ffnGate[layer], ffnUp[layer], ffnDown[layer], allFfnIn, allFfnOut, cfg.isGeGLU(), T);
            }

            if (fusedSuccess) {
                for (int t = 0; t < T; t++) {
                    System.arraycopy(allFfnOut, t * dim, xbL, 0, dim);
                    if (postFfnNorm[layer] != null) {
                        GGUFVectorOps.rmsNorm(xbL, postFfnNorm[layer], xbL, dim, eps);
                    }
                    GGUFVectorOps.addInPlace(X[t], xbL, dim);
                }
            } else {
                for (int t = 0; t < T; t++) {
                    System.arraycopy(allFfnIn, t * dim, xbL, 0, dim);
                    if (ffnGate[layer] != null) {
                        dispatchMatVecPair(ffnGate[layer], gateL, ffnUp[layer], upL, xbL);
                        if (cfg.isGeGLU()) {
                            GGUFVectorOps.geGLU(gateL, upL, hbL, ffnDim);
                        } else {
                            GGUFVectorOps.swiGLU(gateL, upL, hbL, ffnDim);
                        }
                    } else {
                        dispatchMatVec(ffnUp[layer], xbL, upL);
                        GGUFVectorOps.siluInPlace(upL, ffnDim);
                        System.arraycopy(upL, 0, hbL, 0, ffnDim);
                    }
                    dispatchMatVec(ffnDown[layer], hbL, xbL);
                    if (postFfnNorm[layer] != null) {
                        GGUFVectorOps.rmsNorm(xbL, postFfnNorm[layer], xbL, dim, eps);
                    }
                    GGUFVectorOps.addInPlace(X[t], xbL, dim);
                }
            }
            long tFfn = System.nanoTime() - tFfn0;

            // Per-layer embedding + Layer output scale
            for (int t = 0; t < T; t++) {
                if (hasPerLayer && perLayerInpGate[layer] != null && perLayerProj[layer] != null) {
                    dispatchMatVec(perLayerInpGate[layer], X[t], peGate);
                    float kGelu = 0.7978845608f;
                    int peSliceOffset = layer * nEmbdPL;
                    for (int j = 0; j < nEmbdPL; j++) {
                        float g = peGate[j];
                        float gelu = 0.5f * g * (1.0f + (float) Math.tanh(kGelu * (g + 0.044715f * g * g * g)));
                        peGate[j] = gelu * perLayerInputsT[t][peSliceOffset + j];
                    }
                    dispatchMatVec(perLayerProj[layer], peGate, peOut);
                    if (perLayerPostNorm[layer] != null) {
                        GGUFVectorOps.rmsNorm(peOut, perLayerPostNorm[layer], peOut, dim, eps);
                    }
                    GGUFVectorOps.addInPlace(X[t], peOut, dim);
                }

                if (outScale[layer] != 1.0f) {
                    GGUFVectorOps.scale(X[t], outScale[layer], dim);
                }
            }

            if (layer == 0 || layer == 24 || layer == layers - 1) {
                System.out.printf("[PREFILL LAYER %d] qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms (fused=%b)%n",
                        layer, tQkv / 1e6, tAttn / 1e6, tWo / 1e6, tFfn / 1e6, fusedSuccess);
            }
        }

        // Advance KV cache past all prefill tokens
        for (int t = 0; t < T; t++) cache.advance();

        // If Metal decode step is active, sync prompt KV cache to GPU
        if (canUseMetalDecodeStep) {
            for (int l = 0; l < layers; l++) {
                GgufMetalBridge.syncKVCache(l, T, cache.kLayer(l), cache.vLayer(l));
            }
        }

        // ── 4. Output: logits for last token ────────────────────────────────
        float[] xLast = X[T - 1];
        float[] xbFinal = new float[dim];
        GGUFVectorOps.rmsNorm(xLast, outputNorm, xbFinal, dim, cfg.rmsNormEps());

        float[] logits = new float[cfg.vocabSize()];
        dispatchMatVec(outputWeight, xbFinal, logits);

        float cap = cfg.finalLogitSoftcap();
        if (cap > 0.0f) {
            float invCap = 1.0f / cap;
            for (int i = 0; i < logits.length; i++) {
                logits[i] = cap * (float) Math.tanh(logits[i] * invCap);
            }
        }
        return logits;
    }

    /** One token's forward pass with optional logit computation. */
    float[] forward(int tokenId, int pos, KVCache cache, boolean computeLogits) {
        int dim = cfg.embeddingDim();
        int nH = cfg.nHeads();
        int ffnDim = cfg.ffnDim();
        int layers = cfg.nLayers();
        int nEmbdPerLayer = cfg.nEmbdPerLayer();
        float eps = cfg.rmsNormEps();

        if (canUseMetalDecodeStep && computeLogits) {
            GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenId, x);
            if (cfg.embeddingScale() > 0f && cfg.embeddingScale() != 1.0f) {
                GGUFVectorOps.scale(x, cfg.embeddingScale(), dim);
            }
            long t0 = System.nanoTime();
            boolean ok = GgufMetalBridge.decodeStep(pos, cfg.slidingWindow(), eps, x, this.logits);
            if (ok) {
                if (System.getenv("ALJABR_DEBUG_LAYERS") != null && pos == 27) {
                    forward(tokenId, pos, cache, false);
                }
                cache.advance();
                long tElapsed = System.nanoTime() - t0;
                if (pos % 10 == 0 || pos < 5) {
                    System.out.printf("[METAL DECODE %d] single cmd buffer step: %.2f ms%n", pos, tElapsed / 1e6);
                }
                return this.logits;
            }
        }

        GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenId, x);

        // Scale token embeddings when configured (e.g. sqrt(dim) for Gemma, or explicit embedding scale)
        if (cfg.embeddingScale() > 0f && cfg.embeddingScale() != 1.0f) {
            GGUFVectorOps.scale(x, cfg.embeddingScale(), dim);
        }

        // Project per-layer inputs if per_layer_tok_embd is present
        boolean hasPerLayer = perLayerTokEmbd != null && nEmbdPerLayer > 0;
        if (hasPerLayer) {
            // 1. Fetch row from per_layer_token_embd.weight for this tokenId
            GgufTensorOps.dequantizeRow(model, perLayerTokEmbd, tokenId, perLayerRawTok);
            float tokEmbdScale = (float) Math.sqrt(nEmbdPerLayer);
            GGUFVectorOps.scale(perLayerRawTok, tokEmbdScale, nEmbdPerLayer * layers);

            // 2. per_layer_model_proj matvec from inpL (x before layers)
            dispatchMatVec(perLayerModelProj, x, perLayerModelProjOut);
            float projScale = 1.0f / (float) Math.sqrt(dim);
            GGUFVectorOps.scale(perLayerModelProjOut, projScale, nEmbdPerLayer * layers);

            // 3. RMSNorm per_layer_proj_norm per layer slice of length nEmbdPerLayer
            if (perLayerProjNorm != null) {
                for (int l = 0; l < layers; l++) {
                    int off = l * nEmbdPerLayer;
                    GGUFVectorOps.rmsNorm(perLayerModelProjOut, off, perLayerProjNorm, perLayerModelProjOut, off, nEmbdPerLayer, eps);
                }
            }

            // 4. Combine: (per_layer_proj + per_layer_tok_embd) / sqrt(2)
            float inputScale = 1.0f / (float) Math.sqrt(2.0);
            for (int j = 0; j < nEmbdPerLayer * layers; j++) {
                perLayerInputs[j] = (perLayerModelProjOut[j] + perLayerRawTok[j]) * inputScale;
            }
        }

        long tQkv = 0, tRope = 0, tAttn = 0, tWo = 0, tFfn = 0, tLogits = 0;
        boolean nextQkvReady = false;
        boolean logitsComputed = false;
        long tStartTotal = System.nanoTime();
        for (int layer = 0; layer < layers; layer++) {
            int layerNKVH = cfg.nKVHeadsForLayer(layer);
            int layerHDim = cfg.headDimForLayer(layer);
            float layerRopeFreq = cfg.ropeFreqBaseForLayer(layer);
            float[] layerRopeFreqs = ropeFreqs[layer];

            long t0 = System.nanoTime();
            if (!nextQkvReady) {
                GGUFVectorOps.rmsNorm(x, attnNorm[layer], xb, dim, eps);

                // Q / K / V
                if (wK[layer] != null && wV[layer] != null) {
                    dispatchMatVecTriple(wQ[layer], q, wK[layer], k, wV[layer], v, xb);
                } else if (wK[layer] != null) {
                    dispatchMatVecPair(wQ[layer], q, wK[layer], k, xb);
                    System.arraycopy(k, 0, v, 0, layerNKVH * layerHDim);
                } else {
                    dispatchMatVec(wQ[layer], xb, q);
                }
            }
            nextQkvReady = false;
            long tQkvEnd = System.nanoTime();
            tQkv += (tQkvEnd - t0);

            long tRope0 = System.nanoTime();
            // Q head norms before RoPE
            if (attnQNorm[layer] != null) {
                int qNormLen = attnQNorm[layer].length;
                for (int h = 0; h < nH; h++) {
                    GGUFVectorOps.rmsNorm(q, h * layerHDim, attnQNorm[layer], q, h * layerHDim, qNormLen, eps);
                }
            }

            // RoPE on Q
            GGUFVectorOps.rope(q, nH, layerHDim, pos, layerRopeFreq, layerRopeFreqs);

            int kvLayerToUse = layer;
            if (wK[layer] != null) {

                if (attnKNorm[layer] != null) {
                    int kNormLen = attnKNorm[layer].length;
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNorm(k, kvH * layerHDim, attnKNorm[layer], k, kvH * layerHDim, kNormLen, eps);
                    }
                }

                if (attnVNorm[layer] != null) {
                    int vNormLen = attnVNorm[layer].length;
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNorm(v, kvH * layerHDim, attnVNorm[layer], v, kvH * layerHDim, vNormLen, eps);
                    }
                } else if (cfg.hasUnitVNorm()) {
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNormUnit(v, kvH * layerHDim, v, kvH * layerHDim, layerHDim, eps);
                    }
                }

                GGUFVectorOps.rope(k, layerNKVH, layerHDim, pos, layerRopeFreq, layerRopeFreqs);

                cache.store(layer, k, v);
            } else {
                kvLayerToUse = Math.max(0, layer - (cfg.isSwaLayer(layer) ? 2 : 1));
            }
            tRope += (System.nanoTime() - tRope0);

            // Attention scale
            float scale;
            if (cfg.attentionScale() > 0f) {
                scale = cfg.attentionScale();
            } else if (attnQNorm[layer] != null || attnKNorm[layer] != null) {
                scale = 1.0f;   // Q/K norms already control the scale
            } else {
                scale = (float) (1.0 / Math.sqrt(layerHDim));
            }
            int seqLen = pos + 1;

            // Single-token decode: always causal (attend to all stored positions)
            int calculatedStart = 0;
            if (cfg.isSwaLayer(layer) && cfg.slidingWindow() > 0) {
                calculatedStart = Math.max(0, seqLen - cfg.slidingWindow());
            }
            final int tStart = calculatedStart;

            long tAttn0 = System.nanoTime();
            GGUFVectorOps.zero(attnFull, nH * layerHDim);

            int layerStride = cache.headStride(kvLayerToUse);
            float[] kCache = cache.kLayer(kvLayerToUse);
            float[] vCache = cache.vLayer(kvLayerToUse);
            int kHeads = cache.nKVHeads(kvLayerToUse);

            int activeLen = seqLen - tStart;
            for (int h = 0; h < nH; h++) {
                int kvH = h / (nH / kHeads);
                int qOff = h * layerHDim;

                for (int t = tStart; t < seqLen; t++) {
                    int kBase = t * layerStride + kvH * layerHDim;
                    scores[t - tStart] = GGUFVectorOps.dot(q, qOff, kCache, kBase, layerHDim) * scale;
                }

                GGUFVectorOps.softmax(scores, activeLen);

                int hBase = h * layerHDim;
                for (int t = tStart; t < seqLen; t++) {
                    int vBase = t * layerStride + kvH * layerHDim;
                    GGUFVectorOps.scaledAddInPlace(attnFull, hBase, scores[t - tStart], vCache, vBase, layerHDim);
                }
            }
            long tAttnEnd = System.nanoTime();
            tAttn += (tAttnEnd - tAttn0);

            long tWo0 = System.nanoTime();
            boolean woFfnDone = false;
            // Enable next-layer QKV fusion when the next layer has independent KV weights,
            // no per-layer embedding, and is not the last layer.
            boolean canFuseNext = !hasPerLayer
                    && layer + 1 < layers
                    && wQ[layer + 1] != null
                    && wK[layer + 1] != null
                    && wV[layer + 1] != null
                    && attnNorm[layer + 1] != null
                    && GgufMetalBridge.isAvailable();

            if (ffnGate[layer] != null && !hasPerLayer) {
                if (layer == layers - 1 && computeLogits && outputWeight != null && outputNorm != null && GgufMetalBridge.isAvailable()) {
                    float cap = cfg.finalLogitSoftcap();
                    woFfnDone = GgufMetalBridge.woFfnLogitsFused(
                            model, layer, x, attnFull, wO[layer],
                            postAttnNorm[layer], ffnNorm[layer], postFfnNorm[layer],
                            eps, outScale[layer],
                            ffnGate[layer], ffnUp[layer], ffnDown[layer],
                            cfg.isGeGLU(),
                            outputNorm, outputWeight,
                            this.logits, cap);
                    if (woFfnDone) {
                        logitsComputed = true;
                    }
                }
                if (!woFfnDone && canFuseNext) {
                    woFfnDone = GgufMetalBridge.woFfnNextQkvFused(
                            model, layer, x, attnFull, wO[layer],
                            postAttnNorm[layer], ffnNorm[layer], postFfnNorm[layer],
                            eps, outScale[layer],
                            ffnGate[layer], ffnUp[layer], ffnDown[layer],
                            cfg.isGeGLU(),
                            true, attnNorm[layer + 1],
                            q, k, v,
                            wQ[layer + 1], wK[layer + 1], wV[layer + 1]);
                    if (woFfnDone) {
                        nextQkvReady = true;
                    }
                }
                if (!woFfnDone) {
                    woFfnDone = GgufMetalBridge.woFfnFused(
                            model, layer, x, attnFull, wO[layer],
                            postAttnNorm[layer], ffnNorm[layer], postFfnNorm[layer],
                            eps, ffnGate[layer], ffnUp[layer], ffnDown[layer],
                            cfg.isGeGLU());
                }
            }

            if (woFfnDone) {
                long tFusedEnd = System.nanoTime();
                long elapsed = tFusedEnd - tWo0;
                if (logitsComputed) {
                    tWo += elapsed / 4;
                    tFfn += elapsed / 4;
                    tLogits = elapsed / 2;
                } else {
                    tWo += elapsed / 2;
                    tFfn += elapsed / 2;
                }
            } else {
                dispatchMatVec(wO[layer], attnFull, xb);

                // Post-attention norm (if present in model)
                if (postAttnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(xb, postAttnNorm[layer], xb, dim, eps);
                }
                GGUFVectorOps.addInPlace(x, xb, dim);
                tWo += (System.nanoTime() - tWo0);

                long tFfn0 = System.nanoTime();
                GGUFVectorOps.rmsNorm(x, ffnNorm[layer], xb, dim, eps);
                if (ffnGate[layer] != null && dispatchFfnFused(ffnGate[layer], ffnUp[layer], ffnDown[layer], xb, xb, cfg.isGeGLU())) {
                    // Done directly in xb
                } else {
                    if (ffnGate[layer] != null) {
                        dispatchMatVecPair(ffnGate[layer], gate, ffnUp[layer], up, xb);
                        if (cfg.isGeGLU()) {
                            GGUFVectorOps.geGLU(gate, up, hb, ffnDim);
                        } else {
                            GGUFVectorOps.swiGLU(gate, up, hb, ffnDim);
                        }
                    } else {
                        dispatchMatVec(ffnUp[layer], xb, up);
                        GGUFVectorOps.siluInPlace(up, ffnDim);
                        System.arraycopy(up, 0, hb, 0, ffnDim);
                    }
                    dispatchMatVec(ffnDown[layer], hb, xb);
                }

                // Post-FFN norm (if present in model)
                if (postFfnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(xb, postFfnNorm[layer], xb, dim, eps);
                }
                GGUFVectorOps.addInPlace(x, xb, dim);
                tFfn += (System.nanoTime() - tFfn0);
            }

            // Per-layer embedding residual
            if (hasPerLayer && perLayerInpGate[layer] != null && perLayerProj[layer] != null) {
                // peGate = gelu(inp_gate(x))
                dispatchMatVec(perLayerInpGate[layer], x, peGate);
                float kGelu = 0.7978845608f;
                int peSliceOffset = layer * nEmbdPerLayer;
                for (int j = 0; j < nEmbdPerLayer; j++) {
                    float g = peGate[j];
                    float gelu = 0.5f * g * (1.0f + (float) Math.tanh(kGelu * (g + 0.044715f * g * g * g)));
                    // element-wise multiply with inp_this_layer
                    peGate[j] = gelu * perLayerInputs[peSliceOffset + j];
                }
                // peOut = post_norm(proj(peGate))
                dispatchMatVec(perLayerProj[layer], peGate, peOut);
                if (perLayerPostNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(peOut, perLayerPostNorm[layer], peOut, dim, eps);
                }
                GGUFVectorOps.addInPlace(x, peOut, dim);
            }

            // Layer output scale (layer_scalar): applied ONCE to the full layer output x,
            // after all residuals (attention + FFN + per-layer embedding), before next layer.
            // If nextQkvReady is true, outScale was already applied on GPU in woFfnNextQkvFused.
            if (!nextQkvReady && !logitsComputed && outScale[layer] != 1.0f) {
                GGUFVectorOps.scale(x, outScale[layer], dim);
            }
            if (System.getenv("ALJABR_DEBUG_LAYERS") != null && pos == 27) {
                float sumSq = 0f;
                for (float v : x) sumSq += v * v;
                System.out.printf("[JAVA  L%02d] x[0]=%+.4f, norm=%.4f%n", layer, x[0], (float) Math.sqrt(sumSq));
            }
            if (pos == 25 && (layer == 0 || layer == 1 || layer == 24 || layer == layers - 1)) {
                System.out.printf("[LAYER %d TIMING pos=%d] qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms%n",
                        layer, pos, (tQkv) / 1e6, (tAttn) / 1e6, (tWo) / 1e6, (tFfn) / 1e6);
            }
        }

        cache.advance();

        if (computeLogits && !logitsComputed) {
            long tLogits0 = System.nanoTime();
            GGUFVectorOps.rmsNorm(x, outputNorm, xb, dim, eps);
            long tNormEnd = System.nanoTime();

            float cap = cfg.finalLogitSoftcap();
            long tMatVecEnd;
            long tTanhEnd;
            if (cap > 0.0f && GgufMetalBridge.matVecSoftcap(model, outputWeight, xb, this.logits, cap)) {
                tMatVecEnd = System.nanoTime();
                tTanhEnd = tMatVecEnd;
            } else {
                dispatchMatVec(outputWeight, xb, this.logits);
                tMatVecEnd = System.nanoTime();
                if (cap > 0.0f) {
                    // GPU path: aljabr_softcap_kernel (262144 * tanh on GPU ≈ 0.5 ms vs 6-14 ms CPU)
                    boolean gpuDone = GgufMetalBridge.softcapLogits(this.logits, cap);
                    if (!gpuDone) {
                        // CPU fallback — identical numerical result (precise::tanh matches Math.tanh)
                        float invCap = 1.0f / cap;
                        for (int i = 0; i < logits.length; i++) {
                            logits[i] = cap * (float) Math.tanh(logits[i] * invCap);
                        }
                    }
                }
                tTanhEnd = System.nanoTime();
            }
            tLogits = tTanhEnd - tLogits0;
            if (pos <= 2 || pos == 24 || pos == 25) {
                System.out.printf("[LOGITS DETAIL pos=%d] norm=%.2fms, matvec=%.2fms, tanh(cap=%.1f)=%.2fms, total=%.2fms%n",
                        pos, (tNormEnd - tLogits0) / 1e6, (tMatVecEnd - tNormEnd) / 1e6, cap, (tTanhEnd - tMatVecEnd) / 1e6, tLogits / 1e6);
            }
        }

        long tTotal = System.nanoTime() - tStartTotal;
        if (Boolean.getBoolean("aljabr.timing") || pos <= 2 || pos == 24 || pos == 25) {
            System.out.printf("[TIMING pos=%d] total=%.1fms (qkv=%.1fms, rope=%.1fms, attn=%.1fms, wo=%.1fms, ffn=%.1fms, logits=%.1fms)%n",
                    pos, tTotal / 1e6, tQkv / 1e6, tRope / 1e6, tAttn / 1e6, tWo / 1e6, tFfn / 1e6, tLogits / 1e6);
        }

        if (!computeLogits) {
            return null;
        }
        return this.logits;
    }

    /** Overload for backward compatibility computing logits by default. */
    float[] forward(int tokenId, int pos, KVCache cache) {
        return forward(tokenId, pos, cache, true);
    }

    private static long countMetal = 0;
    private static long countCpu = 0;

    private void dispatchMatVec(GGUFTensorInfo tensor, float[] input, float[] output) {
        if (GgufMetalBridge.matVec(model, tensor, input, output)) {
            countMetal++;
        } else {
            countCpu++;
            if (countCpu <= 15) {
                System.err.println("[FALLBACK] tensor " + tensor.name() + " (typeId=" + tensor.typeId() + ", shape=" + java.util.Arrays.toString(tensor.shape()) + ") fell back to CPU!");
            }
            GgufTensorOps.matVecParallel(model, tensor, input, output);
        }
    }

    private void dispatchMatVecPair(GGUFTensorInfo t0, float[] out0, GGUFTensorInfo t1, float[] out1, float[] input) {
        if (GgufMetalBridge.matVecPair(model, t0, out0, t1, out1, input)) {
            countMetal += 2;
        } else {
            dispatchMatVec(t0, input, out0);
            dispatchMatVec(t1, input, out1);
        }
    }

    private void dispatchMatVecTriple(GGUFTensorInfo t0, float[] out0, GGUFTensorInfo t1, float[] out1, GGUFTensorInfo t2, float[] out2, float[] input) {
        if (GgufMetalBridge.matVecTriple(model, t0, out0, t1, out1, t2, out2, input)) {
            countMetal += 3;
        } else {
            dispatchMatVec(t0, input, out0);
            dispatchMatVec(t1, input, out1);
            dispatchMatVec(t2, input, out2);
        }
    }

    private void dispatchMatVecBatch(GGUFTensorInfo tensor, float[] inputBatch, float[] outputBatch, int batchSize) {
        if (GgufMetalBridge.matVecBatch(model, tensor, inputBatch, outputBatch, batchSize)) {
            countMetal += batchSize;
        } else {
            int K = (int) GgufTensorOps.matrixColumns(tensor);
            int N = (int) GgufTensorOps.matrixRows(tensor);
            float[] inSlice = new float[K];
            float[] outSlice = new float[N];
            for (int t = 0; t < batchSize; t++) {
                System.arraycopy(inputBatch, t * K, inSlice, 0, K);
                dispatchMatVec(tensor, inSlice, outSlice);
                System.arraycopy(outSlice, 0, outputBatch, t * N, N);
            }
        }
    }

    private void dispatchMatVecTripleBatch(GGUFTensorInfo t0, float[] out0,
                                           GGUFTensorInfo t1, float[] out1,
                                           GGUFTensorInfo t2, float[] out2,
                                           float[] inputBatch, int batchSize) {
        if (GgufMetalBridge.matVecTripleBatch(model, t0, out0, t1, out1, t2, out2, inputBatch, batchSize)) {
            countMetal += 3L * batchSize;
        } else {
            int K = (int) GgufTensorOps.matrixColumns(t0);
            int N0 = (int) GgufTensorOps.matrixRows(t0);
            int N1 = (int) GgufTensorOps.matrixRows(t1);
            int N2 = (int) GgufTensorOps.matrixRows(t2);
            float[] inSlice = new float[K];
            float[] s0 = new float[N0];
            float[] s1 = new float[N1];
            float[] s2 = new float[N2];
            for (int t = 0; t < batchSize; t++) {
                System.arraycopy(inputBatch, t * K, inSlice, 0, K);
                dispatchMatVecTriple(t0, s0, t1, s1, t2, s2, inSlice);
                System.arraycopy(s0, 0, out0, t * N0, N0);
                System.arraycopy(s1, 0, out1, t * N1, N1);
                System.arraycopy(s2, 0, out2, t * N2, N2);
            }
        }
    }

    private boolean dispatchFfnFused(GGUFTensorInfo gate, GGUFTensorInfo up, GGUFTensorInfo down,
                                     float[] inA, float[] outDown, boolean isGeGLU) {
        if (GgufMetalBridge.ffnGatedFused(model, gate, up, down, inA, outDown, isGeGLU)) {
            countMetal += 3;
            return true;
        }
        return false;
    }

    private boolean dispatchFfnFusedBatch(GGUFTensorInfo gate, GGUFTensorInfo up, GGUFTensorInfo down,
                                          float[] inABatch, float[] outDownBatch, boolean isGeGLU, int batchSize) {
        if (GgufMetalBridge.ffnGatedFusedBatch(model, gate, up, down, inABatch, outDownBatch, isGeGLU, batchSize)) {
            countMetal += 3L * batchSize;
            return true;
        }
        return false;
    }

    private GGUFTensorInfo requireTensor(String name) {
        return findTensor(name)
                .orElseThrow(() -> new IllegalStateException(
                        "GGUF model is missing required tensor '" + name + "' for architecture '"
                                + cfg.architecture() + "'"));
    }

    private Optional<GGUFTensorInfo> findTensor(String name) {
        try {
            return Optional.ofNullable(GgufTensorOps.findTensor(model, name));
        } catch (IllegalArgumentException e) {
            return Optional.empty();
        }
    }

    private float[] dequantizeVector(GGUFTensorInfo tensor, int fallbackDim) {
        int dim = (tensor.shape() != null && tensor.shape().length > 0 && tensor.shape()[0] > 0)
                ? (int) tensor.shape()[0]
                : fallbackDim;
        float[] out = new float[dim];
        GgufTensorOps.dequantizeRow(model, tensor, 0L, out);
        return out;
    }
}
