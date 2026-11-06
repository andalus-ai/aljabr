package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

import java.util.function.BiConsumer;

/**
 * Coordinator for single-token autoregressive decoding passes.
 *
 * <p>Supports full-GPU Metal single command buffer decode step (greedy or with logits),
 * as well as fused/per-op CPU and GPU decode fallbacks.</p>
 */
final class GgufDecodePass {

    private final GGUFModel model;
    private final GgufModelConfig cfg;
    private final GgufLayerWeights layerWeights;
    private final GgufExecutionDispatcher execution;
    private final GGUFTensorInfo tokenEmbed;
    private final GGUFTensorInfo perLayerTokEmbd;
    private final GGUFTensorInfo perLayerModelProj;
    private final float[] perLayerProjNorm;
    private final GGUFTensorInfo outputWeight;
    private final float[] outputNorm;

    private final GgufMetalDecodeState metalDecodeState;
    private final boolean canUseMetalDecodeStep;
    private final boolean canUseMetalTokenEmbd;
    private final BiConsumer<Integer, String> fallbackDisableHandler;

    // Scratch buffers sized to worst-case across layers
    private final float[] x, xb, q, k, v, attnFull, attnOut, gate, up, hb, scores, logits, ffnOutBuf;
    private final float[][] headScores;
    private final float[] perLayerRawTok, perLayerModelProjOut, perLayerInputs;
    private final float[] peGate, peOut, pePostNormBuf;
    private final int maxKVHeads;

    GgufDecodePass(
            GGUFModel model,
            GgufModelConfig cfg,
            GgufLayerWeights layerWeights,
            GgufExecutionDispatcher execution,
            GGUFTensorInfo tokenEmbed,
            GGUFTensorInfo perLayerTokEmbd,
            GGUFTensorInfo perLayerModelProj,
            float[] perLayerProjNorm,
            GGUFTensorInfo outputWeight,
            float[] outputNorm,
            GgufMetalDecodeState metalDecodeState,
            boolean canUseMetalDecodeStep,
            boolean canUseMetalTokenEmbd,
            int maxKVHeads,
            BiConsumer<Integer, String> fallbackDisableHandler) {
        this.model = model;
        this.cfg = cfg;
        this.layerWeights = layerWeights;
        this.execution = execution;
        this.tokenEmbed = tokenEmbed;
        this.perLayerTokEmbd = perLayerTokEmbd;
        this.perLayerModelProj = perLayerModelProj;
        this.perLayerProjNorm = perLayerProjNorm;
        this.outputWeight = outputWeight;
        this.outputNorm = outputNorm;
        this.metalDecodeState = metalDecodeState;
        this.canUseMetalDecodeStep = canUseMetalDecodeStep;
        this.canUseMetalTokenEmbd = canUseMetalTokenEmbd;
        this.maxKVHeads = maxKVHeads;
        this.fallbackDisableHandler = fallbackDisableHandler;

        int dim = cfg.embeddingDim();
        int ffnDim = cfg.ffnDim();
        int nH = cfg.nHeads();
        int hDim = cfg.headDim();
        int layers = cfg.nLayers();
        int nEmbdPerLayer = cfg.nEmbdPerLayer();

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
    }

    boolean canUseMetalGreedy() {
        return metalDecodeState.canUse(canUseMetalDecodeStep);
    }

    float[] scratchPeGate() {
        return peGate;
    }

    float[] scratchPeOut() {
        return peOut;
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
            GgufKvCacheSequenceLifecycle.completeDecode(cache, pos);
            metalDecodeState.recordSuccessfulDecode();
            long tElapsed = System.nanoTime() - t0;
            if (GgufDecodeStep.isVerbose() && (pos % 10 == 0 || pos < 5)) {
                System.out.printf("[METAL DECODE GREEDY %d] single cmd buffer step: %.2f ms%n", pos,
                        tElapsed / 1e6);
            }
            return bestToken;
        }

        String fallbackReason = GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(false,
                metalDecodeState.hasCommitted());
        if (metalDecodeState.recordRuntimeFailure() == GgufMetalDecodeState.RuntimeFailure.CONTEXT_RESTART_REQUIRED) {
            String message = "[GgufDecodeStep] " + fallbackReason
                    + "; cannot safely continue with the incomplete host KV cache. "
                    + "Recreate the decoder and prefill the full context.";
            System.err.println(message);
            throw new IllegalStateException(message);
        }

        if (fallbackDisableHandler != null) {
            fallbackDisableHandler.accept(pos, fallbackReason);
        }
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
                GgufKvCacheSequenceLifecycle.completeDecode(cache, pos);
                metalDecodeState.recordSuccessfulDecode();
                long tElapsed = System.nanoTime() - t0;
                if (GgufDecodeStep.isVerbose() && computeLogits && (pos % 10 == 0 || pos < 5)) {
                    System.out.printf("[METAL DECODE %d] single cmd buffer step: %.2f ms%n", pos,
                            tElapsed / 1e6);
                }
                return this.logits;
            }

            String fallbackReason = GgufDecodeEligibility.fullGpuDecodeRuntimeFailureReason(false,
                    metalDecodeState.hasCommitted());
            if (metalDecodeState
                    .recordRuntimeFailure() == GgufMetalDecodeState.RuntimeFailure.CONTEXT_RESTART_REQUIRED) {
                String message = "[GgufDecodeStep] " + fallbackReason
                        + "; cannot safely continue with the incomplete host KV cache. "
                        + "Recreate the decoder and prefill the full context.";
                System.err.println(message);
                throw new IllegalStateException(message);
            }

            // No generated token has been committed by the native path yet, so the
            // host KV cache still contains all required history and per-op fallback is safe.
            if (fallbackDisableHandler != null) {
                fallbackDisableHandler.accept(pos, fallbackReason);
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
                    execution.dispatchMatVecTriple(layerWeights.wQ[layer], q, layerWeights.wK[layer], k,
                            layerWeights.wV[layer], v, xb);
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
                    GGUFVectorOps.rmsNorm(q, h * layerHDim, layerWeights.attnQNorm[layer], q, h * layerHDim, qNormLen,
                            eps);
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
                        GGUFVectorOps.rmsNorm(k, kvH * layerHDim, layerWeights.attnKNorm[layer], k, kvH * layerHDim,
                                kNormLen, eps);
                    }
                }

                if (layerWeights.attnVNorm[layer] != null) {
                    int vNormLen = layerWeights.attnVNorm[layer].length;
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNorm(v, kvH * layerHDim, layerWeights.attnVNorm[layer], v, kvH * layerHDim,
                                vNormLen, eps);
                    }
                } else if (cfg.hasUnitVNorm()) {
                    for (int kvH = 0; kvH < layerNKVH; kvH++) {
                        GGUFVectorOps.rmsNormUnit(v, kvH * layerHDim, v, kvH * layerHDim, layerHDim, eps);
                    }
                }

                GGUFVectorOps.rope(k, layerNKVH, layerHDim, cfg.rotaryDimForLayer(layer), pos, layerRopeFreq,
                        layerRopeFreqs);

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

            if (GgufDecodeStep.isVerbose() && pos == 27 && layer == 0) {
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
                            layerWeights.postAttnNorm[layer], layerWeights.ffnNorm[layer],
                            layerWeights.postFfnNorm[layer],
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
                            layerWeights.postAttnNorm[layer], layerWeights.ffnNorm[layer],
                            layerWeights.postFfnNorm[layer],
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
                            layerWeights.postAttnNorm[layer], layerWeights.ffnNorm[layer],
                            layerWeights.postFfnNorm[layer],
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
                        && execution.dispatchFfnFused(layerWeights.ffnGate[layer], layerWeights.ffnUp[layer],
                                layerWeights.ffnDown[layer], xb, xb, cfg.isGeGLU())) {
                    // Done directly in xb
                } else {
                    if (layerWeights.ffnGate[layer] != null) {
                        execution.dispatchMatVecPair(layerWeights.ffnGate[layer], gate, layerWeights.ffnUp[layer], up,
                                xb);
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
            if (hasPerLayer && layerWeights.perLayerInpGate[layer] != null
                    && layerWeights.perLayerProj[layer] != null) {
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
            // after all residuals (attention + FFN + per-layer embedding), before next layer.
            // If nextQkvReady is true, layerWeights.outScale was already applied on GPU in woFfnNextQkvFused.
            if (!nextQkvReady && !logitsComputed && layerWeights.outScale[layer] != 1.0f) {
                GGUFVectorOps.scale(x, layerWeights.outScale[layer], dim);
            }
            if (System.getenv("ALJABR_DEBUG_LAYERS") != null && pos == 27) {
                float sumSq = 0f;
                for (float v : x)
                    sumSq += v * v;
                System.out.printf("[JAVA  L%02d] x[0]=%+.4f, norm=%.4f%n", layer, x[0], (float) Math.sqrt(sumSq));
            }
            if (GgufDecodeStep.isVerbose() && pos == 25 && (layer == 0 || layer == 1 || layer == 24 || layer == layers - 1)) {
                System.out.printf("[LAYER %d TIMING pos=%d] qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms%n",
                        layer, pos, (tQkv) / 1e6, (tAttn) / 1e6, (tWo) / 1e6, (tFfn) / 1e6);
            }
        }

        GgufKvCacheSequenceLifecycle.completeDecode(cache, pos);

        if (computeLogits && !logitsComputed) {
            float cap = cfg.finalLogitSoftcap();
            GgufLogitsProcessor.Timing timing = GgufLogitsProcessor.compute(
                    model, outputWeight, outputNorm, x, xb, this.logits, dim, eps, cap, execution);
            tLogits = timing.totalNanos();
            if (GgufDecodeStep.isVerbose() && (pos <= 2 || pos == 24 || pos == 25)) {
                System.out.printf(
                        "[LOGITS DETAIL pos=%d] norm=%.2fms, matvec=%.2fms, tanh(cap=%.1f)=%.2fms, total=%.2fms%n",
                        pos, timing.normalizationNanos() / 1e6, timing.projectionNanos() / 1e6, cap,
                        timing.softcapNanos() / 1e6, tLogits / 1e6);
            }
        }

        long tTotal = System.nanoTime() - tStartTotal;
        if ((Boolean.getBoolean("aljabr.timing") || (GgufDecodeStep.isVerbose() && (pos <= 2 || pos == 24 || pos == 25)))) {
            System.out.printf(
                    "[TIMING pos=%d] total=%.1fms (qkv=%.1fms, rope=%.1fms, attn=%.1fms, wo=%.1fms, ffn=%.1fms, logits=%.1fms)%n",
                    pos, tTotal / 1e6, tQkv / 1e6, tRope / 1e6, tAttn / 1e6, tWo / 1e6, tFfn / 1e6, tLogits / 1e6);
        }

        if (GgufDecodeStep.isVerbose() && pos == 27) {
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
}
