package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.inference.KVCache;

/** Copies shared KV-cache entries into the contiguous batch buffers used by prefill attention. */
final class GgufKvCacheReadExecution {
    private GgufKvCacheReadExecution() {
    }

    /**
     * Copies {@code tokenCount} entries from one cache layer into contiguous
     * key/value batch buffers with {@code targetStride} values per token.
     *
     * <p>The source cache may have a wider per-token stride than the attention
     * view. In that case only the leading {@code targetStride} values are
     * copied from each token, preserving the existing shared-KV layout.</p>
     */
    static void copyLayerToBatch(
            KVCache cache,
            int sourceLayer,
            float[] targetKeys,
            float[] targetValues,
            int tokenCount,
            int targetStride) {
        if (cache == null) {
            throw new IllegalArgumentException("KV cache must not be null");
        }
        if (tokenCount < 0 || tokenCount > cache.seqLen()) {
            throw new IllegalArgumentException("tokenCount must be within the committed KV sequence");
        }
        int sourceStride = cache.headStride(sourceLayer);
        if (targetStride <= 0 || targetStride > sourceStride) {
            throw new IllegalArgumentException("targetStride must be positive and no wider than the source layer stride");
        }
        int targetLength;
        try {
            targetLength = Math.multiplyExact(tokenCount, targetStride);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("KV batch target size exceeds supported array size", e);
        }
        if (targetKeys == null || targetValues == null
                || targetKeys.length < targetLength || targetValues.length < targetLength) {
            throw new IllegalArgumentException("KV batch target arrays are shorter than the requested copy");
        }
        float[] sourceKeys = cache.kLayer(sourceLayer);
        float[] sourceValues = cache.vLayer(sourceLayer);
        if (sourceStride == targetStride) {
            System.arraycopy(sourceKeys, 0, targetKeys, 0, tokenCount * targetStride);
            System.arraycopy(sourceValues, 0, targetValues, 0, tokenCount * targetStride);
        } else {
            for (int token = 0; token < tokenCount; token++) {
                int sourceOffset = token * sourceStride;
                int targetOffset = token * targetStride;
                System.arraycopy(sourceKeys, sourceOffset, targetKeys, targetOffset, targetStride);
                System.arraycopy(sourceValues, sourceOffset, targetValues, targetOffset, targetStride);
            }
        }
    }
}
