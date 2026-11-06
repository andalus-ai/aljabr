package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import java.util.Locale;
import java.util.function.BiConsumer;

/**
 * Coordinator for model prefill (prompt processing).
 *
 * <p>Coordinates embedding preparation, per-layer projections, batched QKV projection,
 * attention execution, residual and FFN execution, KV cache advancement,
 * and final logits generation.</p>
 */
final class GgufPrefillPass {

    private final GGUFModel model;
    private final GGUFTensorInfo tokenEmbed;
    private final GGUFTensorInfo perLayerTokEmbd;
    private final GGUFTensorInfo perLayerModelProj;
    private final float[] perLayerProjNorm;
    private final float[] peGate;
    private final float[] peOut;
    private final GGUFTensorInfo outputWeight;
    private final float[] outputNorm;
    private final GgufModelConfig cfg;
    private final GgufLayerWeights layerWeights;
    private final GgufExecutionDispatcher execution;
    private final GgufMetalDecodeState metalDecodeState;
    private final boolean canUseMetalDecodeStep;
    private final int maxKVHeads;
    private final BiConsumer<Integer, String> fallbackDisableHandler;

    GgufPrefillPass(
            GGUFModel model,
            GGUFTensorInfo tokenEmbed,
            GGUFTensorInfo perLayerTokEmbd,
            GGUFTensorInfo perLayerModelProj,
            float[] perLayerProjNorm,
            float[] peGate,
            float[] peOut,
            GGUFTensorInfo outputWeight,
            float[] outputNorm,
            GgufModelConfig cfg,
            GgufLayerWeights layerWeights,
            GgufExecutionDispatcher execution,
            GgufMetalDecodeState metalDecodeState,
            boolean canUseMetalDecodeStep,
            int maxKVHeads,
            BiConsumer<Integer, String> fallbackDisableHandler) {
        this.model = model;
        this.tokenEmbed = tokenEmbed;
        this.perLayerTokEmbd = perLayerTokEmbd;
        this.perLayerModelProj = perLayerModelProj;
        this.perLayerProjNorm = perLayerProjNorm;
        this.peGate = peGate;
        this.peOut = peOut;
        this.outputWeight = outputWeight;
        this.outputNorm = outputNorm;
        this.cfg = cfg;
        this.layerWeights = layerWeights;
        this.execution = execution;
        this.metalDecodeState = metalDecodeState;
        this.canUseMetalDecodeStep = canUseMetalDecodeStep;
        this.maxKVHeads = maxKVHeads;
        this.fallbackDisableHandler = fallbackDisableHandler;
    }

    float[] execute(int[] tokenIds, KVCache cache) {
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

        // Reusable batch buffers across layers (0 allocations inside layer loop)
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

        // Opt-in aggregate profiling: reports the whole prefill pass and distinguishes attention backend use.
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
            long tQkvWall0 = System.nanoTime();
            long tQkv = GgufPrefillQkvProjection.project(
                    X, allXb, allQ, allK, allV, T, layer, layerWeights, cfg, execution,
                    cache, hasOwnKv, kvLayerToUse, attentionKvDim);
            long tQkvWallMs = (System.nanoTime() - tQkvWall0) / 1_000_000L;
            if (tQkvWallMs > 50) { // per-layer QKV > 50ms is a spike
                System.err.printf("[QKV SPIKE] layer=%d wall=%dms dispatch=%.1fms%n",
                        layer, tQkvWallMs, tQkv / 1e6);
            }
            if (profilePrefill)
                profileQkvNanos += tQkv;

            // ── 3b. Attention for all T tokens ───────────────────────────────
            int swWin = cfg.slidingWindow();
            long tAttn0 = System.nanoTime();
            boolean gpuAttnDone = GgufPrefillAttentionExecution.execute(
                    allAttn, allQ, allK, allV, attnScores, T, nH, attentionKVHeads,
                    layerHDim, scale, isSwa, swWin, hasOwnKv, kvDim, attentionKvDim);
            long tAttn = System.nanoTime() - tAttn0;
            if (profilePrefill)
                profileAttentionNanos += tAttn;
            if (profilePrefill && gpuAttnDone) {
                profileGpuAttentionLayers++;
            }

            // ── 3c. Output projection + residual + FFN for all T tokens ──────
            GgufPrefillResidualExecution.Result residualResult = GgufPrefillResidualExecution.execute(
                    model, layer, X, allAttn, allWoOut, allFfnIn, allFfnOut,
                    xbL, gateL, upL, hbL, peGate, peOut, perLayerInputsT, hasPerLayer,
                    T, dim, qDim, nEmbdPL, eps, layerWeights, cfg, execution,
                    metalDecodeState, canUseMetalDecodeStep);

            if (profilePrefill) {
                if (residualResult.fusedLayerDone()) {
                    profileFusedLayers++;
                    profileFusedLayerNanos += residualResult.elapsedNanos();
                } else {
                    profileWoNanos += residualResult.tWoNanos();
                    profileFfnNanos += residualResult.tFfnNanos();
                }
            }

            if (GgufDecodeStep.isVerbose() && (layer == 0 || layer == 24 || layer == layers - 1)) {
                if (residualResult.fusedLayerDone()) {
                    System.out.printf("[PREFILL LAYER %d] qkv=%.2fms attn=%.2fms wo_ffn_fused=%.2fms%n",
                            layer, tQkv / 1e6, tAttn / 1e6, residualResult.elapsedNanos() / 1e6);
                } else {
                    System.out.printf("[PREFILL LAYER %d] qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms (fused=%b)%n",
                            layer, tQkv / 1e6, tAttn / 1e6, residualResult.tWoNanos() / 1e6,
                            residualResult.tFfnNanos() / 1e6, residualResult.fusedSuccess());
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
        GgufKvCacheSequenceLifecycle.completePrefill(cache, T);

        // Prompt KV synchronization is a one-time handoff from the populated host cache.
        if (metalDecodeState.canUse(canUseMetalDecodeStep)) {
            int failedLayer = GgufPrefillMetalCacheSync.sync(
                    T, layerWeights.wK, cfg, cache, layers);
            if (failedLayer >= 0) {
                String reason = GgufDecodeEligibility.fullGpuDecodeKvCacheSyncFailureReason(failedLayer);
                if (fallbackDisableHandler != null) {
                    fallbackDisableHandler.accept(T, reason);
                }
            }
        }

        // ── 4. Output: logits for last token ────────────────────────────────
        float[] xbFinal = new float[dim];
        float[] logits = new float[cfg.vocabSize()];
        GgufLogitsProcessor.computePrefillSlice(
                model, outputWeight, outputNorm, X, (T - 1) * dim,
                xbFinal, logits, dim, cfg.rmsNormEps(), cfg.finalLogitSoftcap(), execution);

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

        if (GgufDecodeStep.isVerbose()) {
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
}
