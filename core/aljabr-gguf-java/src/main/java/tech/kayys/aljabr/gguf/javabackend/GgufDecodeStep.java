package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * One decoder forward pass: {@code token, position -> logits}, given a
 * running {@link KVCache}.
 *
 * <p>
 * Supports standard LLaMA/Mistral/Qwen models as well as Gemma family models
 * (Gemma 2, Gemma 3, Gemma 4) with embedding scaling, per-head Q/K/V RMSNorm,
 * post-attention and post-FFN RMSNorm, GeGLU activations, per-layer SWA / RoPE,
 * rope_freqs proportional rotary embedding, per-layer token embeddings, layer
 * output scaling, and final logit softcapping.
 * </p>
 *
 * <p>
 * Gemma 4 SWA layers use <em>bidirectional</em> attention during prefill
 * (all tokens in the prompt attend to each other within the sliding window),
 * matching the {@code LLAMA_NON_CAUSAL_TYPE_SWA_ONLY} behaviour in llama.cpp.
 * Use {@link #prefill(int[], KVCache)} for prompt processing and
 * {@link #forward(int, int, KVCache, boolean)} for single-token decode.
 * </p>
 */
final class GgufDecodeStep {
    private static final Logger LOG = Logger.getLogger(GgufDecodeStep.class.getName());

    public static boolean isVerbose() {
        return Boolean.getBoolean("aljabr.metal.verbose")
                || Boolean.getBoolean("aqli.verbose")
                || System.getenv("ALJABR_METAL_VERBOSE") != null
                || System.getenv("ALJABR_DEBUG") != null
                || System.getenv("AQLI_DEBUG") != null
                || LOG.isLoggable(Level.FINE);
    }

    private final GGUFModel model;
    private final GgufExecutionDispatcher execution;
    private final GgufModelConfig cfg;

    private final GgufLayerWeights layerWeights;
    private final GgufTensorResolver tensorResolver;
    private final GGUFTensorInfo tokenEmbed;
    private final float[] outputNorm;
    private final GGUFTensorInfo outputWeight;

    // Per-layer token embeddings (Gemma 4)
    private final GGUFTensorInfo perLayerTokEmbd;
    private final GGUFTensorInfo perLayerModelProj;
    private final float[] perLayerProjNorm;



    // Scratch buffers sized to worst-case (maximum) across layers
    private final float[] x, xb, q, k, v, attnFull, attnOut, gate, up, hb, scores, logits, ffnOutBuf;
    private final float[][] headScores;
    private final float[] perLayerRawTok, perLayerModelProjOut, perLayerInputs;
    private final float[] peGate, peOut, pePostNormBuf;

    // Max KV heads across all layers (real max, not clamped to nH)
    private final int maxKVHeads;

    // Full-GPU single command buffer decode step available
    private final boolean canUseMetalDecodeStep;
    private final boolean canUseMetalTokenEmbd;

    private final GgufMetalDecodeState metalDecodeState = new GgufMetalDecodeState();

    /**
     * A key projection means this layer owns its KV entries. Gemma 4 can omit
     * {@code attn_v.weight}; in that case the projected key is reused as the
     * value input, but the layer still owns its KV cache. Do not misclassify
     * such layers as cross-layer KV-sharing layers during prefill.
     */


    /**
     * Explains why the native full-GPU decode path cannot be used before any
     * native buffers are registered. Kept pure so the eligibility policy can
     * be regression-tested without a GPU or a model file.
     */
    /**
     * Returns whether a model's KV projection layout is not safe for the
     * current native full-GPU decoder. Independent K/V projections, K-as-V
     * layers, and shared-KV owners with matching geometry are supported. A
     * V-only layer is not representable by the native decode path.
     *
     * <p>Array entries are per-layer; a shared-KV layer with both projections
     * absent uses the preceding one layer for global attention or two layers
     * back for SWA, matching the current registration convention.</p>
     */








    /**
     * Classifies a failed native decode call. Falling back is safe only before
     * the native path has successfully advanced any generated-token KV entries,
     * because the host KV cache is not populated by successful native steps.
     */


    /**
     * Prompt KV synchronization happens before generation and before the native
     * decoder has committed any token. If any layer fails to sync, the populated
     * host cache remains the safe source of truth for per-op decoding.
     */


    GgufDecodeStep(GGUFModel model, GgufModelConfig cfg) {
        this.model = model;
        this.execution = new GgufExecutionDispatcher(model);
        this.cfg = cfg;
        this.tensorResolver = new GgufTensorResolver(model, cfg.architecture());

        int dim = cfg.embeddingDim();
        int ffnDim = cfg.ffnDim();
        int nH = cfg.nHeads();
        int hDim = cfg.headDim();
        int layers = cfg.nLayers();
        int nEmbdPerLayer = cfg.nEmbdPerLayer();

        // Compute true max KV heads across all layers (ignore nKVHeads which may be
        // wrong)
        int maxKVH = 1;
        if (cfg.nKVHeadsPerLayer() != null) {
            for (int kvh : cfg.nKVHeadsPerLayer())
                maxKVH = Math.max(maxKVH, kvh);
        } else {
            maxKVH = cfg.nKVHeads();
        }
        this.maxKVHeads = maxKVH;

        this.tokenEmbed = tensorResolver.requireTensor("token_embd.weight");
        this.outputNorm = tensorResolver.dequantizeVector(tensorResolver.requireTensor("output_norm.weight"), dim);
        this.outputWeight = tensorResolver.findTensor("output.weight").orElse(tokenEmbed); // tied embeddings

        // Per-layer token embedding tensors
        this.perLayerTokEmbd = tensorResolver.findTensor("per_layer_token_embd.weight").orElse(null);
        this.perLayerModelProj = tensorResolver.findTensor("per_layer_model_proj.weight").orElse(null);
        this.perLayerProjNorm = tensorResolver.findTensor("per_layer_proj_norm.weight")
                .map(t -> tensorResolver.dequantizeVector(t, nEmbdPerLayer > 0 ? nEmbdPerLayer : 1))
                .orElse(null);

        
        
        

        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        

        this.layerWeights = GgufLayerWeightResolver.resolve(
                tensorResolver, cfg, dim, nEmbdPerLayer, perLayerTokEmbd);

        // Opt-in layer-shape diagnostics verify architecture-specific RoPE dimensions
        // on the actual loaded GGUF, especially for Gemma 4's local/global attention mix.
        if (System.getenv("ALJABR_DEBUG_LAYERS") != null) {
            int previousHeadDim = -1;
            int previousRotaryDim = -1;
            boolean previousSwa = false;
            for (int layer = 0; layer < layers; layer++) {
                int layerHeadDim = cfg.headDimForLayer(layer);
                int rotaryDim = cfg.rotaryDimForLayer(layer);
                boolean swa = cfg.isSwaLayer(layer);
                if (isVerbose() && (layer == 0 || layerHeadDim != previousHeadDim
                        || rotaryDim != previousRotaryDim || swa != previousSwa)) {
                    System.out.printf(
                            "[JAVA ROPE CONFIG] layer=%d attention=%s head_dim=%d rotary_dim=%d rope_base=%.1f freq_factors=%d%n",
                            layer, swa ? "swa" : "global", layerHeadDim, rotaryDim,
                            cfg.ropeFreqBaseForLayer(layer),
                            layerWeights.ropeFreqs[layer] == null ? 0 : layerWeights.ropeFreqs[layer].length);
                }
                previousHeadDim = layerHeadDim;
                previousRotaryDim = rotaryDim;
                previousSwa = swa;
            }
        }

        // Register static layer norms on GPU bridge once to enable zero-copy decode
        // slices
        if (GgufMetalBridge.isAvailable()) {
            GgufMetalBridge.registerLayerNorms(layers, dim, hDim, layerWeights.postAttnNorm, layerWeights.ffnNorm, layerWeights.postFfnNorm, layerWeights.attnNorm,
                    layerWeights.attnQNorm, layerWeights.attnKNorm);
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

        GgufMetalDecodeRegistration.Result metalRegistration =
                GgufMetalDecodeRegistration.register(model, cfg, layerWeights, tokenEmbed, perLayerTokEmbd,
                        outputNorm, outputWeight, dim, ffnDim, nH, layers, nEmbdPerLayer);
        this.canUseMetalDecodeStep = metalRegistration.decodeActive();
        this.canUseMetalTokenEmbd = metalRegistration.tokenEmbeddingRegistered();
        if (isVerbose()) {
            if (metalRegistration.decodeActive()) {
                System.out.println("[GgufDecodeStep] Full-GPU single command buffer decode step ENABLED (" + layers + " layers).");
            } else {
                String reason = metalRegistration.skipReason() == null
                        ? "unknown eligibility or registration failure" : metalRegistration.skipReason();
                System.out.println("[GgufDecodeStep] Full-GPU single command buffer decode step UNAVAILABLE: "
                        + reason + "; using the existing per-op decode path.");
            }
        }
    }

    /**
     * Batch-prefill all prompt tokens through the network in one shot.
     *
     * <p>
     * For Gemma 4 SWA layers this uses <em>bidirectional</em> attention
     * within the sliding window (matching llama.cpp's
     * {@code LLAMA_NON_CAUSAL_TYPE_SWA_ONLY}). Global (dense) layers remain
     * strictly causal. All KV pairs are deposited into {@code cache} so that
     * subsequent single-token decode calls can build on the prefill.
     *
     * @param tokenIds prompt token IDs, length T
     * @param cache    KV cache to populate (must be empty / positioned at 0)
     * @return logits array for the last token
     */
    float[] prefill(int[] tokenIds, KVCache cache) {
        int T = tokenIds.length;
        int dim = cfg.embeddingDim();
        int nH = cfg.nHeads();
        int ffnDim = cfg.ffnDim();
        int layers = cfg.nLayers();
        int nEmbdPL = cfg.nEmbdPerLayer();
        float eps = cfg.rmsNormEps();
        float embScale = cfg.embeddingScale();

        // ── 1. Token embeddings: X[T * dim] flat continuous buffer ────────────
        float[] X = GgufPrefillEmbeddingPreparation.prepareTokenEmbeddings(
                model, tokenEmbed, tokenIds, dim, embScale);

        // ── 2. Per-layer token embedding projection (architecture optional) ──
        float[][] perLayerInputsT = GgufPrefillPerLayerEmbeddingPreparation.prepare(
                model, perLayerTokEmbd, perLayerModelProj, perLayerProjNorm,
                tokenIds, X, dim, nEmbdPL, layers, eps, execution);
        boolean hasPerLayer = perLayerInputsT != null;

        // ── 3. Layer loop ─────────────────────────────────────────────────────
        int maxHDim = cfg.headDim();

        // Reusable batch buffers across layers (0 allocations inside 48-layer loop)
        float[] allXb = new float[T * dim];
        float[] allQ = new float[T * nH * maxHDim];
        float[] allK = new float[T * maxKVHeads * maxHDim];
        float[] allV = new float[T * maxKVHeads * maxHDim];
        float[] allAttn = new float[T * nH * maxHDim];
        float[] allWoOut = new float[T * dim];
        float[] allFfnIn = new float[T * dim];
        float[] allFfnOut = new float[T * dim];
        float[] xbL = new float[dim];
        float[] gateL = new float[ffnDim];
        float[] upL = new float[ffnDim];
        float[] hbL = new float[ffnDim];
        float[] attnScores = new float[T];

        // Opt-in aggregate profiling: unlike the sampled layer logs above, this
        // reports the whole prefill pass and distinguishes attention backend use.
        boolean profilePrefill = Boolean.getBoolean("aljabr.metal.prefill.profile")
                || "1".equals(System.getenv("ALJABR_METAL_PREFILL_PROFILE"));
        long profileQkvNanos = 0L;
        long profileAttentionNanos = 0L;
        long profileWoNanos = 0L;
        long profileFfnNanos = 0L;
        long profileFusedLayerNanos = 0L;
        int profileGpuAttentionLayers = 0;
        int profileFusedLayers = 0;
        long profileLayerLoopStart = System.nanoTime();

        for (int layer = 0; layer < layers; layer++) {
            int layerNKVH = cfg.nKVHeadsForLayer(layer);
            int layerHDim = cfg.headDimForLayer(layer);
            float layerRopeFreq = cfg.ropeFreqBaseForLayer(layer);
            float[] layerRopeFreqs = layerWeights.ropeFreqs[layer];
            boolean isSwa = cfg.isSwaLayer(layer);

            float scale;
            if (cfg.attentionScale() > 0f) {
                scale = cfg.attentionScale();
            } else if (layerWeights.attnQNorm[layer] != null || layerWeights.attnKNorm[layer] != null) {
                scale = 1.0f;
            } else {
                scale = (float) (1.0 / Math.sqrt(layerHDim));
            }

            int qDim = nH * layerHDim;
            int kvDim = layerNKVH * layerHDim;
            boolean hasOwnKv = GgufDecodeEligibility.hasOwnKvCache(layerWeights.wK[layer] != null);
            int kvLayerToUse = GgufKvCacheRouting.resolveCacheLayer(layer, hasOwnKv, isSwa);
            int attentionKVHeads = hasOwnKv ? layerNKVH : cache.nKVHeads(kvLayerToUse);
            int attentionKvDim = attentionKVHeads * layerHDim;

            // ── 3a. QKV projections for all T tokens ─────────────────────────
            for (int t = 0; t < T; t++) {
                GGUFVectorOps.rmsNorm(X, t * dim, layerWeights.attnNorm[layer], allXb, t * dim, dim, eps);
            }

            long tQkv0 = System.nanoTime();
            if (layerWeights.wK[layer] != null && layerWeights.wV[layer] != null) {
                execution.dispatchMatVecTripleBatch(layerWeights.wQ[layer], allQ, layerWeights.wK[layer], allK, layerWeights.wV[layer], allV, allXb, T);
            } else if (layerWeights.wK[layer] != null) {
                execution.dispatchMatVecBatch(layerWeights.wQ[layer], allXb, allQ, T);
                execution.dispatchMatVecBatch(layerWeights.wK[layer], allXb, allK, T);
                System.arraycopy(allK, 0, allV, 0, T * kvDim);
            } else {
                execution.dispatchMatVecBatch(layerWeights.wQ[layer], allXb, allQ, T);
            }
            long tQkv = System.nanoTime() - tQkv0;
            if (profilePrefill) profileQkvNanos += tQkv;

            for (int t = 0; t < T; t++) {
                if (layerWeights.attnQNorm[layer] != null) {
                    int qNormLen = layerWeights.attnQNorm[layer].length;
                    for (int h = 0; h < nH; h++) {
                        int off = t * qDim + h * layerHDim;
                        GGUFVectorOps.rmsNorm(allQ, off, layerWeights.attnQNorm[layer], allQ, off, qNormLen, eps);
                    }
                }
                GGUFVectorOps.rope(allQ, t * qDim, nH, layerHDim, cfg.rotaryDimForLayer(layer), t, layerRopeFreq, layerRopeFreqs);

                if (hasOwnKv) {
                    if (layerWeights.attnKNorm[layer] != null) {
                        int kNormLen = layerWeights.attnKNorm[layer].length;
                        for (int kvH = 0; kvH < layerNKVH; kvH++) {
                            int off = t * kvDim + kvH * layerHDim;
                            GGUFVectorOps.rmsNorm(allK, off, layerWeights.attnKNorm[layer], allK, off, kNormLen, eps);
                        }
                    }
                    if (layerWeights.attnVNorm[layer] != null) {
                        int vNormLen = layerWeights.attnVNorm[layer].length;
                        for (int kvH = 0; kvH < layerNKVH; kvH++) {
                            int off = t * kvDim + kvH * layerHDim;
                            GGUFVectorOps.rmsNorm(allV, off, layerWeights.attnVNorm[layer], allV, off, vNormLen, eps);
                        }
                    } else if (cfg.hasUnitVNorm()) {
                        for (int kvH = 0; kvH < layerNKVH; kvH++) {
                            int off = t * kvDim + kvH * layerHDim;
                            GGUFVectorOps.rmsNormUnit(allV, off, allV, off, layerHDim, eps);
                        }
                    }

                    GGUFVectorOps.rope(allK, t * kvDim, layerNKVH, layerHDim, cfg.rotaryDimForLayer(layer), t, layerRopeFreq, layerRopeFreqs);
                }
            }

            if (hasOwnKv) {
                GgufKvCacheWriteExecution.storeBatch(cache, layer, 0, allK, 0, allV, 0, T * kvDim);
            } else {
                float[] sourceK = cache.kLayer(kvLayerToUse);
                float[] sourceV = cache.vLayer(kvLayerToUse);
                int sourceStride = cache.headStride(kvLayerToUse);
                if (sourceStride == attentionKvDim) {
                    System.arraycopy(sourceK, 0, allK, 0, T * attentionKvDim);
                    System.arraycopy(sourceV, 0, allV, 0, T * attentionKvDim);
                } else {
                    for (int t = 0; t < T; t++) {
                        int sourceOffset = t * sourceStride;
                        System.arraycopy(sourceK, sourceOffset, allK, t * attentionKvDim, attentionKvDim);
                        System.arraycopy(sourceV, sourceOffset, allV, t * attentionKvDim, attentionKvDim);
                    }
                }
            }

            // ── 3b. Attention for all T tokens ───────────────────────────────
            int swWin = cfg.slidingWindow();
            long tAttn0 = System.nanoTime();
            boolean gpuAttnDone = GgufPrefillAttentionExecution.execute(
                    allAttn, allQ, allK, allV, attnScores, T, nH, attentionKVHeads,
                    layerHDim, scale, isSwa, swWin, hasOwnKv, kvDim, attentionKvDim);
            long tAttn = System.nanoTime() - tAttn0;
            if (profilePrefill) profileAttentionNanos += tAttn;
            if (profilePrefill && gpuAttnDone) {
                profileGpuAttentionLayers++;
            }

            // ── 3c. Output projection + residual + FFN for all T tokens ──────
            long tWo0 = System.nanoTime();
            boolean fusedLayerDone = false;
            long tWo = 0;
            long tFfn = 0;
            boolean fusedSuccess = false;

            if (metalDecodeState.canUse(canUseMetalDecodeStep) && !hasPerLayer) {
                fusedLayerDone = GgufMetalBridge.woFfnFusedLayerBatch(
                        model, layer, X, allAttn, dim, qDim, T, eps, cfg.isGeGLU());
            }

            if (!fusedLayerDone) {
                execution.dispatchMatVecBatch(layerWeights.wO[layer], allAttn, allWoOut, T);

                for (int t = 0; t < T; t++) {
                    System.arraycopy(allWoOut, t * dim, xbL, 0, dim);
                    if (layerWeights.postAttnNorm[layer] != null) {
                        GGUFVectorOps.rmsNorm(xbL, layerWeights.postAttnNorm[layer], xbL, dim, eps);
                    }
                    GGUFVectorOps.addInPlace(X, t * dim, xbL, 0, dim);
                }
                tWo = System.nanoTime() - tWo0;

                // FFN
                long tFfn0 = System.nanoTime();
                for (int t = 0; t < T; t++) {
                    GGUFVectorOps.rmsNorm(X, t * dim, layerWeights.ffnNorm[layer], allFfnIn, t * dim, dim, eps);
                }

                if (layerWeights.ffnGate[layer] != null) {
                    fusedSuccess = execution.dispatchFfnFusedBatch(layerWeights.ffnGate[layer], layerWeights.ffnUp[layer], layerWeights.ffnDown[layer], allFfnIn, allFfnOut,
                            cfg.isGeGLU(), T);
                }

                if (fusedSuccess) {
                    for (int t = 0; t < T; t++) {
                        System.arraycopy(allFfnOut, t * dim, xbL, 0, dim);
                        if (layerWeights.postFfnNorm[layer] != null) {
                            GGUFVectorOps.rmsNorm(xbL, layerWeights.postFfnNorm[layer], xbL, dim, eps);
                        }
                        GGUFVectorOps.addInPlace(X, t * dim, xbL, 0, dim);
                    }
                } else {
                    for (int t = 0; t < T; t++) {
                        System.arraycopy(allFfnIn, t * dim, xbL, 0, dim);
                        if (layerWeights.ffnGate[layer] != null) {
                            execution.dispatchMatVecPair(layerWeights.ffnGate[layer], gateL, layerWeights.ffnUp[layer], upL, xbL);
                            if (cfg.isGeGLU()) {
                                GGUFVectorOps.geGLU(gateL, upL, hbL, ffnDim);
                            } else {
                                GGUFVectorOps.swiGLU(gateL, upL, hbL, ffnDim);
                            }
                        } else {
                            execution.dispatchMatVec(layerWeights.ffnUp[layer], xbL, upL);
                            GGUFVectorOps.siluInPlace(upL, ffnDim);
                            System.arraycopy(upL, 0, hbL, 0, ffnDim);
                        }
                        execution.dispatchMatVec(layerWeights.ffnDown[layer], hbL, xbL);
                        if (layerWeights.postFfnNorm[layer] != null) {
                            GGUFVectorOps.rmsNorm(xbL, layerWeights.postFfnNorm[layer], xbL, dim, eps);
                        }
                        GGUFVectorOps.addInPlace(X, t * dim, xbL, 0, dim);
                    }
                }
                tFfn = System.nanoTime() - tFfn0;
                if (profilePrefill) {
                    profileWoNanos += tWo;
                    profileFfnNanos += tFfn;
                }

                // Per-layer embedding + Layer output scale
                for (int t = 0; t < T; t++) {
                    if (hasPerLayer && layerWeights.perLayerInpGate[layer] != null && layerWeights.perLayerProj[layer] != null) {
                        float[] xSlice = new float[dim];
                        System.arraycopy(X, t * dim, xSlice, 0, dim);
                        execution.dispatchMatVec(layerWeights.perLayerInpGate[layer], xSlice, peGate);
                        float kGelu = 0.7978845608f;
                        int peSliceOffset = layer * nEmbdPL;
                        for (int j = 0; j < nEmbdPL; j++) {
                            float g = peGate[j];
                            float gelu = 0.5f * g * (1.0f + (float) Math.tanh(kGelu * (g + 0.044715f * g * g * g)));
                            peGate[j] = gelu * perLayerInputsT[t][peSliceOffset + j];
                        }
                        execution.dispatchMatVec(layerWeights.perLayerProj[layer], peGate, peOut);
                        if (layerWeights.perLayerPostNorm[layer] != null) {
                            GGUFVectorOps.rmsNorm(peOut, layerWeights.perLayerPostNorm[layer], peOut, dim, eps);
                        }
                        GGUFVectorOps.addInPlace(X, t * dim, peOut, 0, dim);
                    }

                    if (layerWeights.outScale[layer] != 1.0f) {
                        GGUFVectorOps.scale(X, t * dim, layerWeights.outScale[layer], dim);
                    }
                }
            }

            if (profilePrefill && fusedLayerDone) {
                profileFusedLayers++;
                profileFusedLayerNanos += System.nanoTime() - tWo0;
            }

            if (isVerbose() && (layer == 0 || layer == 24 || layer == layers - 1)) {
                if (fusedLayerDone) {
                    System.out.printf("[PREFILL LAYER %d] qkv=%.2fms attn=%.2fms wo_ffn_fused=%.2fms%n",
                            layer, tQkv / 1e6, tAttn / 1e6, (System.nanoTime() - tWo0) / 1e6);
                } else {
                    System.out.printf("[PREFILL LAYER %d] qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms (fused=%b)%n",
                            layer, tQkv / 1e6, tAttn / 1e6, tWo / 1e6, tFfn / 1e6, fusedSuccess);
                }
            }
        }

        if (profilePrefill) {
            long layerLoopNanos = System.nanoTime() - profileLayerLoopStart;
            System.out.printf(
                    "[PREFILL PROFILE] tokens=%d layers=%d layer_loop=%.2fms qkv=%.2fms attention=%.2fms "
                            + "attention_backend=metal_fa4:%d/java_or_fallback:%d wo=%.2fms ffn=%.2fms "
                            + "wo_ffn_fused_layers=%d fused_layer_time=%.2fms%n",
                    T, layers, layerLoopNanos / 1e6, profileQkvNanos / 1e6,
                    profileAttentionNanos / 1e6, profileGpuAttentionLayers,
                    layers - profileGpuAttentionLayers, profileWoNanos / 1e6,
                    profileFfnNanos / 1e6, profileFusedLayers, profileFusedLayerNanos / 1e6);
        }

        // Advance KV cache past all prefill tokens
        cache.advance(T);

        // Prompt KV synchronization is a one-time handoff from the populated host cache.
        if (metalDecodeState.canUse(canUseMetalDecodeStep)) {
            int failedLayer = GgufPrefillMetalCacheSync.sync(
                    T, layerWeights.wK, cfg, cache, layers);
            if (failedLayer >= 0) {
                String reason = GgufDecodeEligibility.fullGpuDecodeKvCacheSyncFailureReason(failedLayer);
                disableMetalDecodeAfterRuntimeFallback(T, reason);
            }
        }

        // ── 4. Output: logits for last token ────────────────────────────────
        float[] xbFinal = new float[dim];
        float[] logits = new float[cfg.vocabSize()];
        GgufLogitsProcessor.computePrefillSlice(
                model, outputWeight, outputNorm, X, (T - 1) * dim,
                xbFinal, logits, dim, cfg.rmsNormEps(), cfg.finalLogitSoftcap(), execution);
        // The first generated token is sampled from these prefill logits before
        // forward() attempts the native single-token decode step. Keep a direct,
        // opt-in diagnostic here so bad first-token distributions are not
        // mistakenly attributed to the following native decode call.
        if (System.getenv("ALJABR_DEBUG_LAYERS") != null) {
            float[] xLast = new float[dim];
            System.arraycopy(X, (T - 1) * dim, xLast, 0, dim);
            logPrefillLogits(T, xLast, xbFinal, logits);
        }
        return logits;
    }

    private static void logPrefillLogits(
            int promptTokens, float[] lastHidden, float[] normalized, float[] logits) {
        double hiddenNormSq = 0.0;
        double normalizedNormSq = 0.0;
        int nonFiniteHidden = 0;
        int nonFiniteNormalized = 0;
        int nonFiniteLogits = 0;
        for (float value : lastHidden) {
            if (Float.isFinite(value)) {
                hiddenNormSq += (double) value * value;
            } else {
                nonFiniteHidden++;
            }
        }
        for (float value : normalized) {
            if (Float.isFinite(value)) {
                normalizedNormSq += (double) value * value;
            } else {
                nonFiniteNormalized++;
            }
        }

        final int topCount = Math.min(8, logits.length);
        int[] topIds = new int[topCount];
        float[] topScores = new float[topCount];
        java.util.Arrays.fill(topIds, -1);
        java.util.Arrays.fill(topScores, Float.NEGATIVE_INFINITY);
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            float score = logits[tokenId];
            if (!Float.isFinite(score)) {
                nonFiniteLogits++;
                continue;
            }
            for (int slot = 0; slot < topCount; slot++) {
                if (score > topScores[slot]) {
                    for (int move = topCount - 1; move > slot; move--) {
                        topScores[move] = topScores[move - 1];
                        topIds[move] = topIds[move - 1];
                    }
                    topScores[slot] = score;
                    topIds[slot] = tokenId;
                    break;
                }
            }
        }

        if (isVerbose()) {
            StringBuilder top = new StringBuilder("[");
            for (int i = 0; i < topCount; i++) {
                if (i > 0) {
                    top.append(", ");
                }
                top.append(topIds[i]).append(":")
                        .append(String.format(Locale.ROOT, "%.5f", topScores[i]));
            }
            top.append("]");
            System.out.printf(Locale.ROOT,
                    "[JAVA PREFILL LOGITS] prompt_tokens=%d hidden_norm=%.5f normalized_norm=%.5f nonfinite_hidden=%d nonfinite_normalized=%d nonfinite_logits=%d top=%s%n",
                    promptTokens, Math.sqrt(hiddenNormSq), Math.sqrt(normalizedNormSq),
                    nonFiniteHidden, nonFiniteNormalized, nonFiniteLogits, top);
        }
    }

    private void disableMetalDecodeAfterRuntimeFallback(int pos, String reason) {
        metalDecodeState.disableBeforeCommit();
        System.err.printf(
                "[GgufDecodeStep] Full-GPU decode disabled after position %d: %s; "
                        + "falling back to per-op decode for this and subsequent tokens.%n",
                pos, reason);
    }

    public boolean canUseMetalGreedy() {
        return metalDecodeState.canUse(canUseMetalDecodeStep);
    }

    /**
     * Fast-path forward pass for greedy decoding.
     * Evaluates the model on GPU, computes vDSP argmax directly on GPU logits,
     * and returns the best token without copying vocabulary logits to host memory.
     */
    int forwardGreedy(int tokenId, int pos, KVCache cache) {
        if (!metalDecodeState.canUse(canUseMetalDecodeStep)) {
            return -1;
        }
        if (metalDecodeState.requiresRestart()) {
            throw new IllegalStateException(
                    "Full-GPU Metal decode failed after advancing the native KV cache. "
                            + "Recreate the decoder and prefill the full context before retrying.");
        }
        int dim = cfg.embeddingDim();
        float eps = cfg.rmsNormEps();

        long t0 = System.nanoTime();
        int bestToken = -1;
        if (canUseMetalTokenEmbd) {
            bestToken = GgufMetalBridge.decodeStepTokenGreedy(pos, cfg.slidingWindow(), eps, tokenId);
        }
        if (bestToken < 0) {
            GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenId, x);
            if (cfg.embeddingScale() > 0f && cfg.embeddingScale() != 1.0f) {
                GGUFVectorOps.scale(x, cfg.embeddingScale(), dim);
            }
            t0 = System.nanoTime();
            bestToken = GgufMetalBridge.decodeStepGreedy(pos, cfg.slidingWindow(), eps, x);
        }
        if (bestToken >= 0) {
            cache.advance();
            metalDecodeState.recordSuccessfulDecode();
            long tElapsed = System.nanoTime() - t0;
            if (isVerbose() && (pos % 10 == 0 || pos < 5)) {
                System.out.printf("[METAL DECODE GREEDY %d] single cmd buffer step: %.2f ms%n", pos,
                        tElapsed / 1e6);
            }
            return bestToken;
        }

        String fallbackReason = GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(false, metalDecodeState.hasCommitted());
        if (metalDecodeState.recordRuntimeFailure() == GgufMetalDecodeState.RuntimeFailure.CONTEXT_RESTART_REQUIRED) {
            String message = "[GgufDecodeStep] " + fallbackReason
                    + "; cannot safely continue with the incomplete host KV cache. "
                    + "Recreate the decoder and prefill the full context.";
            System.err.println(message);
            throw new IllegalStateException(message);
        }

        disableMetalDecodeAfterRuntimeFallback(pos, fallbackReason);
        return -1;
    }

    /** One token's forward pass with optional logit computation. */
    float[] forward(int tokenId, int pos, KVCache cache, boolean computeLogits) {
        int dim = cfg.embeddingDim();
        int nH = cfg.nHeads();
        int ffnDim = cfg.ffnDim();
        int layers = cfg.nLayers();
        int nEmbdPerLayer = cfg.nEmbdPerLayer();
        float eps = cfg.rmsNormEps();

        if (metalDecodeState.requiresRestart()) {
            throw new IllegalStateException(
                    "Full-GPU Metal decode failed after advancing the native KV cache. "
                            + "Recreate the decoder and prefill the full context before retrying.");
        }

        if (metalDecodeState.canUse(canUseMetalDecodeStep)) {
            long t0 = System.nanoTime();
            boolean ok = false;
            if (canUseMetalTokenEmbd) {
                ok = GgufMetalBridge.decodeStepToken(pos, cfg.slidingWindow(), eps, tokenId, this.logits);
            }
            if (!ok) {
                GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenId, x);
                if (cfg.embeddingScale() > 0f && cfg.embeddingScale() != 1.0f) {
                    GGUFVectorOps.scale(x, cfg.embeddingScale(), dim);
                }
                t0 = System.nanoTime();
                ok = GgufMetalBridge.decodeStep(pos, cfg.slidingWindow(), eps, x, this.logits);
            }
            if (ok) {
                cache.advance();
                metalDecodeState.recordSuccessfulDecode();
                long tElapsed = System.nanoTime() - t0;
                if (isVerbose() && computeLogits && (pos % 10 == 0 || pos < 5)) {
                    System.out.printf("[METAL DECODE %d] single cmd buffer step: %.2f ms%n", pos,
                            tElapsed / 1e6);
                }
                return this.logits;
            }

            String fallbackReason = GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(false, metalDecodeState.hasCommitted());
            if (metalDecodeState.recordRuntimeFailure() == GgufMetalDecodeState.RuntimeFailure.CONTEXT_RESTART_REQUIRED) {
                String message = "[GgufDecodeStep] " + fallbackReason
                        + "; cannot safely continue with the incomplete host KV cache. "
                        + "Recreate the decoder and prefill the full context.";
                System.err.println(message);
                throw new IllegalStateException(message);
            }

            // No generated token has been committed by the native path yet, so the
            // host KV cache still contains all required history and per-op fallback is safe.
            disableMetalDecodeAfterRuntimeFallback(pos, fallbackReason);
        }

        GgufTensorOps.dequantizeRow(model, tokenEmbed, tokenId, x);

        // Scale token embeddings when configured (e.g. sqrt(dim) for Gemma, or explicit
        // embedding scale)
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
            execution.dispatchMatVec(perLayerModelProj, x, perLayerModelProjOut);
            float projScale = 1.0f / (float) Math.sqrt(dim);
            GGUFVectorOps.scale(perLayerModelProjOut, projScale, nEmbdPerLayer * layers);

            // 3. RMSNorm per_layer_proj_norm per layer slice of length nEmbdPerLayer
            if (perLayerProjNorm != null) {
                for (int l = 0; l < layers; l++) {
                    int off = l * nEmbdPerLayer;
                    GGUFVectorOps.rmsNorm(perLayerModelProjOut, off, perLayerProjNorm, perLayerModelProjOut, off,
                            nEmbdPerLayer, eps);
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
            float[] layerRopeFreqs = layerWeights.ropeFreqs[layer];

            long t0 = System.nanoTime();
            if (!nextQkvReady) {
                GGUFVectorOps.rmsNorm(x, layerWeights.attnNorm[layer], xb, dim, eps);

                // Q / K / V
                if (layerWeights.wK[layer] != null && layerWeights.wV[layer] != null) {
                    execution.dispatchMatVecTriple(layerWeights.wQ[layer], q, layerWeights.wK[layer], k, layerWeights.wV[layer], v, xb);
                } else if (layerWeights.wK[layer] != null) {
                    execution.dispatchMatVecPair(layerWeights.wQ[layer], q, layerWeights.wK[layer], k, xb);
                    System.arraycopy(k, 0, v, 0, layerNKVH * layerHDim);
                } else {
                    execution.dispatchMatVec(layerWeights.wQ[layer], xb, q);
                }
            }
            nextQkvReady = false;
            long tQkvEnd = System.nanoTime();
            tQkv += (tQkvEnd - t0);

            long tRope0 = System.nanoTime();
            // Q head norms before RoPE
            if (layerWeights.attnQNorm[layer] != null) {
                int qNormLen = layerWeights.attnQNorm[layer].length;
                for (int h = 0; h < nH; h++) {
                    GGUFVectorOps.rmsNorm(q, h * layerHDim, layerWeights.attnQNorm[layer], q, h * layerHDim, qNormLen, eps);
                }
            }

            // RoPE on Q
            GGUFVectorOps.rope(q, nH, layerHDim, cfg.rotaryDimForLayer(layer), pos, layerRopeFreq, layerRopeFreqs);

            int kvLayerToUse = GgufKvCacheRouting.resolveCacheLayer(
                    layer, layerWeights.wK[layer] != null, cfg.isSwaLayer(layer));
            if (layerWeights.wK[layer] != null) {

                if (layerWeights.attnKNorm[layer] != null) {
                    int kNormLen = layerWeights.attnKNorm[layer].length;
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNorm(k, kvH * layerHDim, layerWeights.attnKNorm[layer], k, kvH * layerHDim, kNormLen, eps);
                    }
                }

                if (layerWeights.attnVNorm[layer] != null) {
                    int vNormLen = layerWeights.attnVNorm[layer].length;
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNorm(v, kvH * layerHDim, layerWeights.attnVNorm[layer], v, kvH * layerHDim, vNormLen, eps);
                    }
                } else if (cfg.hasUnitVNorm()) {
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNormUnit(v, kvH * layerHDim, v, kvH * layerHDim, layerHDim, eps);
                    }
                }

                GGUFVectorOps.rope(k, layerNKVH, layerHDim, cfg.rotaryDimForLayer(layer), pos, layerRopeFreq, layerRopeFreqs);

                GgufKvCacheWriteExecution.storeToken(cache, layer, k, v);
            }
            tRope += (System.nanoTime() - tRope0);

            // Attention scale
            float scale;
            if (cfg.attentionScale() > 0f) {
                scale = cfg.attentionScale();
            } else if (layerWeights.attnQNorm[layer] != null || layerWeights.attnKNorm[layer] != null) {
                scale = 1.0f; // Q/K norms already control the scale
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
            GgufDecodeAttentionExecution.execute(
                    q, attnFull, scores,
                    cache.kLayer(kvLayerToUse), cache.vLayer(kvLayerToUse),
                    cache.headStride(kvLayerToUse), cache.nKVHeads(kvLayerToUse),
                    nH, layerHDim, seqLen, tStart, scale);
            long tAttnEnd = System.nanoTime();
            tAttn += (tAttnEnd - tAttn0);

            if (isVerbose() && pos == 27 && layer == 0) {
                float qNorm = 0, kNorm = 0, vNorm = 0, aNorm = 0;
                for (float f : q)
                    qNorm += f * f;
                for (float f : k)
                    kNorm += f * f;
                for (float f : v)
                    vNorm += f * f;
                for (float f : attnFull)
                    aNorm += f * f;
                System.out.printf(
                        "[JAVA L00] q[0]=%+.4f norm=%.4f | k[0]=%+.4f norm=%.4f | v[0]=%+.4f norm=%.4f | attn[0]=%+.4f norm=%.4f%n",
                        q[0], (float) Math.sqrt(qNorm), k[0], (float) Math.sqrt(kNorm), v[0], (float) Math.sqrt(vNorm),
                        attnFull[0], (float) Math.sqrt(aNorm));
            }

            long tWo0 = System.nanoTime();
            boolean woFfnDone = false;
            // Enable next-layer QKV fusion when the next layer has independent KV weights,
            // no per-layer embedding, and is not the last layer.
            boolean canFuseNext = !hasPerLayer
                    && layer + 1 < layers
                    && layerWeights.wQ[layer + 1] != null
                    && layerWeights.wK[layer + 1] != null
                    && layerWeights.wV[layer + 1] != null
                    && layerWeights.attnNorm[layer + 1] != null
                    && GgufMetalBridge.isAvailable();

            if (layerWeights.ffnGate[layer] != null && !hasPerLayer) {
                if (layer == layers - 1 && computeLogits && outputWeight != null && outputNorm != null
                        && GgufMetalBridge.isAvailable()) {
                    float cap = cfg.finalLogitSoftcap();
                    woFfnDone = GgufMetalBridge.woFfnLogitsFused(
                            model, layer, x, attnFull, layerWeights.wO[layer],
                            layerWeights.postAttnNorm[layer], layerWeights.ffnNorm[layer], layerWeights.postFfnNorm[layer],
                            eps, layerWeights.outScale[layer],
                            layerWeights.ffnGate[layer], layerWeights.ffnUp[layer], layerWeights.ffnDown[layer],
                            cfg.isGeGLU(),
                            outputNorm, outputWeight,
                            this.logits, cap);
                    if (woFfnDone) {
                        logitsComputed = true;
                    }
                }
                if (!woFfnDone && canFuseNext) {
                    woFfnDone = GgufMetalBridge.woFfnNextQkvFused(
                            model, layer, x, attnFull, layerWeights.wO[layer],
                            layerWeights.postAttnNorm[layer], layerWeights.ffnNorm[layer], layerWeights.postFfnNorm[layer],
                            eps, layerWeights.outScale[layer],
                            layerWeights.ffnGate[layer], layerWeights.ffnUp[layer], layerWeights.ffnDown[layer],
                            cfg.isGeGLU(),
                            true, layerWeights.attnNorm[layer + 1],
                            q, k, v,
                            layerWeights.wQ[layer + 1], layerWeights.wK[layer + 1], layerWeights.wV[layer + 1]);
                    if (woFfnDone) {
                        nextQkvReady = true;
                    }
                }
                if (!woFfnDone) {
                    woFfnDone = GgufMetalBridge.woFfnFused(
                            model, layer, x, attnFull, layerWeights.wO[layer],
                            layerWeights.postAttnNorm[layer], layerWeights.ffnNorm[layer], layerWeights.postFfnNorm[layer],
                            eps, layerWeights.ffnGate[layer], layerWeights.ffnUp[layer], layerWeights.ffnDown[layer],
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
                execution.dispatchMatVec(layerWeights.wO[layer], attnFull, xb);

                // Post-attention norm (if present in model)
                if (layerWeights.postAttnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(xb, layerWeights.postAttnNorm[layer], xb, dim, eps);
                }
                GGUFVectorOps.addInPlace(x, xb, dim);
                tWo += (System.nanoTime() - tWo0);

                long tFfn0 = System.nanoTime();
                GGUFVectorOps.rmsNorm(x, layerWeights.ffnNorm[layer], xb, dim, eps);
                if (layerWeights.ffnGate[layer] != null
                        && execution.dispatchFfnFused(layerWeights.ffnGate[layer], layerWeights.ffnUp[layer], layerWeights.ffnDown[layer], xb, xb, cfg.isGeGLU())) {
                    // Done directly in xb
                } else {
                    if (layerWeights.ffnGate[layer] != null) {
                        execution.dispatchMatVecPair(layerWeights.ffnGate[layer], gate, layerWeights.ffnUp[layer], up, xb);
                        if (cfg.isGeGLU()) {
                            GGUFVectorOps.geGLU(gate, up, hb, ffnDim);
                        } else {
                            GGUFVectorOps.swiGLU(gate, up, hb, ffnDim);
                        }
                    } else {
                        execution.dispatchMatVec(layerWeights.ffnUp[layer], xb, up);
                        GGUFVectorOps.siluInPlace(up, ffnDim);
                        System.arraycopy(up, 0, hb, 0, ffnDim);
                    }
                    execution.dispatchMatVec(layerWeights.ffnDown[layer], hb, xb);
                }

                // Post-FFN norm (if present in model)
                if (layerWeights.postFfnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(xb, layerWeights.postFfnNorm[layer], xb, dim, eps);
                }
                GGUFVectorOps.addInPlace(x, xb, dim);
                tFfn += (System.nanoTime() - tFfn0);
            }

            // Per-layer embedding residual
            if (hasPerLayer && layerWeights.perLayerInpGate[layer] != null && layerWeights.perLayerProj[layer] != null) {
                // peGate = gelu(inp_gate(x))
                execution.dispatchMatVec(layerWeights.perLayerInpGate[layer], x, peGate);
                float kGelu = 0.7978845608f;
                int peSliceOffset = layer * nEmbdPerLayer;
                for (int j = 0; j < nEmbdPerLayer; j++) {
                    float g = peGate[j];
                    float gelu = 0.5f * g * (1.0f + (float) Math.tanh(kGelu * (g + 0.044715f * g * g * g)));
                    // element-wise multiply with inp_this_layer
                    peGate[j] = gelu * perLayerInputs[peSliceOffset + j];
                }
                // peOut = post_norm(proj(peGate))
                execution.dispatchMatVec(layerWeights.perLayerProj[layer], peGate, peOut);
                if (layerWeights.perLayerPostNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(peOut, layerWeights.perLayerPostNorm[layer], peOut, dim, eps);
                }
                GGUFVectorOps.addInPlace(x, peOut, dim);
            }

            // Layer output scale (layer_scalar): applied ONCE to the full layer output x,
            // after all residuals (attention + FFN + per-layer embedding), before next
            // layer.
            // If nextQkvReady is true, layerWeights.outScale was already applied on GPU in
            // woFfnNextQkvFused.
            if (!nextQkvReady && !logitsComputed && layerWeights.outScale[layer] != 1.0f) {
                GGUFVectorOps.scale(x, layerWeights.outScale[layer], dim);
            }
            if (System.getenv("ALJABR_DEBUG_LAYERS") != null && pos == 27) {
                float sumSq = 0f;
                for (float v : x)
                    sumSq += v * v;
                System.out.printf("[JAVA  L%02d] x[0]=%+.4f, norm=%.4f%n", layer, x[0], (float) Math.sqrt(sumSq));
            }
            if (isVerbose() && pos == 25 && (layer == 0 || layer == 1 || layer == 24 || layer == layers - 1)) {
                System.out.printf("[LAYER %d TIMING pos=%d] qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms%n",
                        layer, pos, (tQkv) / 1e6, (tAttn) / 1e6, (tWo) / 1e6, (tFfn) / 1e6);
            }
        }

        cache.advance();

        if (computeLogits && !logitsComputed) {
            float cap = cfg.finalLogitSoftcap();
            GgufLogitsProcessor.Timing timing = GgufLogitsProcessor.compute(
                    model, outputWeight, outputNorm, x, xb, this.logits, dim, eps, cap, execution);
            tLogits = timing.totalNanos();
            if (isVerbose() && (pos <= 2 || pos == 24 || pos == 25)) {
                System.out.printf(
                        "[LOGITS DETAIL pos=%d] norm=%.2fms, matvec=%.2fms, tanh(cap=%.1f)=%.2fms, total=%.2fms%n",
                        pos, timing.normalizationNanos() / 1e6, timing.projectionNanos() / 1e6, cap,
                        timing.softcapNanos() / 1e6, tLogits / 1e6);
            }
        }

        long tTotal = System.nanoTime() - tStartTotal;
        if ((Boolean.getBoolean("aljabr.timing") || (isVerbose() && (pos <= 2 || pos == 24 || pos == 25)))) {
            System.out.printf(
                    "[TIMING pos=%d] total=%.1fms (qkv=%.1fms, rope=%.1fms, attn=%.1fms, wo=%.1fms, ffn=%.1fms, logits=%.1fms)%n",
                    pos, tTotal / 1e6, tQkv / 1e6, tRope / 1e6, tAttn / 1e6, tWo / 1e6, tFfn / 1e6, tLogits / 1e6);
        }

        if (isVerbose() && pos == 27) {
            int best = 0;
            for (int i = 1; i < this.logits.length; i++)
                if (this.logits[i] > this.logits[best])
                    best = i;
            System.out.printf(
                    "[JAVA LOGITS pos=27] argmax=%d score=%.4f (target 236828 score=%.4f, 236772 score=%.4f)%n",
                    best, this.logits[best], this.logits[236828], this.logits[236772]);
        }
        return this.logits;
    }

    /** Overload for backward compatibility computing logits by default. */
    float[] forward(int tokenId, int pos, KVCache cache) {
        return forward(tokenId, pos, cache, true);
    }

    /**
     * A layer is eligible for full-Metal decode only when its layer metadata and
     * every required optional RoPE buffer were registered successfully.
     */
}
