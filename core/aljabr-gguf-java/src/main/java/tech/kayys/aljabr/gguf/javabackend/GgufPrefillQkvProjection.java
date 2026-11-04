package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/**
 * Orchestrates prefill QKV normalization, matrix projection dispatch,
 * per-head RMSNorm, RoPE, and KV cache storage or shared-KV loading.
 */
final class GgufPrefillQkvProjection {
    private GgufPrefillQkvProjection() {
    }

    /**
     * Executes the QKV projection and post-processing for all prompt tokens in a layer.
     *
     * @return elapsed nanoseconds spent specifically in matrix-vector projection dispatch
     */
    static long project(
            float[] hiddenStates,
            float[] normalizedHidden,
            float[] queries,
            float[] keys,
            float[] values,
            int tokenCount,
            int layer,
            GgufLayerWeights layerWeights,
            GgufModelConfig config,
            GgufExecutionDispatcher execution,
            KVCache cache,
            boolean hasOwnKv,
            int kvLayerToUse,
            int attentionKvDimension) {
        int dimension = config.embeddingDim();
        int queryHeads = config.nHeads();
        int headDimension = config.headDimForLayer(layer);
        int kvHeads = config.nKVHeadsForLayer(layer);
        int queryDimension = queryHeads * headDimension;
        int kvDimension = kvHeads * headDimension;
        float rmsNormEpsilon = config.rmsNormEps();
        float ropeFreqBase = config.ropeFreqBaseForLayer(layer);
        float[] ropeFreqFactors = layerWeights.ropeFreqs[layer];
        int rotaryDimension = config.rotaryDimForLayer(layer);

        // 1. RMSNorm attn_norm for all tokens into normalizedHidden
        for (int token = 0; token < tokenCount; token++) {
            GGUFVectorOps.rmsNorm(hiddenStates, token * dimension, layerWeights.attnNorm[layer],
                    normalizedHidden, token * dimension, dimension, rmsNormEpsilon);
        }

        // 2. Dispatch Q, K, V matrix-vector multiplications
        long dispatchStart = System.nanoTime();
        if (layerWeights.wK[layer] != null && layerWeights.wV[layer] != null) {
            execution.dispatchMatVecTripleBatch(
                    layerWeights.wQ[layer], queries,
                    layerWeights.wK[layer], keys,
                    layerWeights.wV[layer], values,
                    normalizedHidden, tokenCount);
        } else if (layerWeights.wK[layer] != null) {
            execution.dispatchMatVecBatch(layerWeights.wQ[layer], normalizedHidden, queries, tokenCount);
            execution.dispatchMatVecBatch(layerWeights.wK[layer], normalizedHidden, keys, tokenCount);
            System.arraycopy(keys, 0, values, 0, tokenCount * kvDimension);
        } else {
            execution.dispatchMatVecBatch(layerWeights.wQ[layer], normalizedHidden, queries, tokenCount);
        }
        long dispatchNanos = System.nanoTime() - dispatchStart;

        // 3. Per-head Q / K / V normalization and RoPE
        for (int token = 0; token < tokenCount; token++) {
            if (layerWeights.attnQNorm[layer] != null) {
                int qNormLen = layerWeights.attnQNorm[layer].length;
                for (int head = 0; head < queryHeads; head++) {
                    int offset = token * queryDimension + head * headDimension;
                    GGUFVectorOps.rmsNorm(queries, offset, layerWeights.attnQNorm[layer],
                            queries, offset, qNormLen, rmsNormEpsilon);
                }
            }
            GGUFVectorOps.rope(queries, token * queryDimension, queryHeads, headDimension,
                    rotaryDimension, token, ropeFreqBase, ropeFreqFactors);

            if (hasOwnKv) {
                if (layerWeights.attnKNorm[layer] != null) {
                    int kNormLen = layerWeights.attnKNorm[layer].length;
                    for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
                        int offset = token * kvDimension + kvHead * headDimension;
                        GGUFVectorOps.rmsNorm(keys, offset, layerWeights.attnKNorm[layer],
                                keys, offset, kNormLen, rmsNormEpsilon);
                    }
                }
                if (layerWeights.attnVNorm[layer] != null) {
                    int vNormLen = layerWeights.attnVNorm[layer].length;
                    for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
                        int offset = token * kvDimension + kvHead * headDimension;
                        GGUFVectorOps.rmsNorm(values, offset, layerWeights.attnVNorm[layer],
                                values, offset, vNormLen, rmsNormEpsilon);
                    }
                } else if (config.hasUnitVNorm()) {
                    for (int kvHead = 0; kvHead < kvHeads; kvHead++) {
                        int offset = token * kvDimension + kvHead * headDimension;
                        GGUFVectorOps.rmsNormUnit(values, offset, values, offset,
                                headDimension, rmsNormEpsilon);
                    }
                }

                GGUFVectorOps.rope(keys, token * kvDimension, kvHeads, headDimension,
                        rotaryDimension, token, ropeFreqBase, ropeFreqFactors);
            }
        }

        // 4. Store to KV cache if own KV, or load from earlier layer if shared KV
        if (hasOwnKv) {
            GgufKvCacheWriteExecution.storeBatch(cache, layer, 0, keys, 0, values, 0,
                    tokenCount * kvDimension);
        } else {
            GgufKvCacheReadExecution.copyLayerToBatch(cache, kvLayerToUse, keys, values,
                    tokenCount, attentionKvDimension);
        }

        return dispatchNanos;
    }
}
