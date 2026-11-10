package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/** Executes causal single-token attention over the active KV-cache window. */
final class GgufDecodeAttentionExecution {
    private GgufDecodeAttentionExecution() {
    }

    /**
     * Computes grouped-query attention for one decode position.
     *
     * <p>The caller selects the KV-cache layer and active sequence window. This
     * keeps cache ownership, SWA layer selection, and KV writes in the decode
     * orchestrator while this class owns only score, softmax, and value
     * accumulation.</p>
     */
    static void execute(
            float[] query,
            float[] output,
            float[] scores,
            float[] keyCache,
            float[] valueCache,
            int cacheHeadStride,
            int kvHeads,
            int queryHeads,
            int headDimension,
            int sequenceLength,
            int startPosition,
            float scale) {
        GGUFVectorOps.zero(output, queryHeads * headDimension);

        int activeLength = sequenceLength - startPosition;
        for (int head = 0; head < queryHeads; head++) {
            int kvHead = head / (queryHeads / kvHeads);
            int queryOffset = head * headDimension;

            for (int position = startPosition; position < sequenceLength; position++) {
                int keyOffset = position * cacheHeadStride + kvHead * headDimension;
                scores[position - startPosition] = GGUFVectorOps.dot(
                        query, queryOffset, keyCache, keyOffset, headDimension) * scale;
            }

            GGUFVectorOps.softmax(scores, activeLength);

            int outputOffset = head * headDimension;
            for (int position = startPosition; position < sequenceLength; position++) {
                int valueOffset = position * cacheHeadStride + kvHead * headDimension;
                GGUFVectorOps.scaledAddInPlace(
                        output, outputOffset, scores[position - startPosition],
                        valueCache, valueOffset, headDimension);
            }
        }
    }
}
