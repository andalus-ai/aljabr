package tech.kayys.aljabr.gguf.javabackend;

import java.util.Arrays;

import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/** Executes causal or sliding-window attention for a prompt prefill batch. */
final class GgufPrefillAttentionExecution {
    private GgufPrefillAttentionExecution() {
    }

    /**
     * Executes prefill attention, preferring the Metal FlashAttention backend when eligible.
     *
     * @return {@code true} when Metal handled attention; {@code false} when the Java fallback ran
     */
    static boolean execute(
            float[] output,
            float[] queries,
            float[] keys,
            float[] values,
            float[] attentionScores,
            int tokenCount,
            int queryHeads,
            int kvHeads,
            int headDimension,
            float scale,
            boolean slidingWindowAttention,
            int slidingWindow,
            boolean hasOwnKv,
            int kvDimension,
            int attentionKvDimension) {
        boolean gpuDone = false;
        if ((!slidingWindowAttention || slidingWindow <= 0 || tokenCount <= slidingWindow)
                && GgufMetalBridge.isAvailable()) {
            gpuDone = GgufMetalBridge.fa4Attention(
                    output, queries, keys, values, 1, tokenCount, tokenCount,
                    queryHeads, kvHeads, headDimension, scale, true);
        }

        if (gpuDone) {
            return true;
        }

        int kvStride = hasOwnKv ? kvDimension : attentionKvDimension;
        for (int token = 0; token < tokenCount; token++) {
            int start;
            int end;
            if (slidingWindowAttention && slidingWindow > 0) {
                start = Math.max(0, token - slidingWindow + 1);
                end = token;
            } else {
                start = 0;
                end = token;
            }
            int activeLength = end - start + 1;

            for (int head = 0; head < queryHeads; head++) {
                int kvHead = head / (queryHeads / kvHeads);
                int queryOffset = token * queryHeads * headDimension + head * headDimension;

                for (int position = start; position <= end; position++) {
                    float dot = 0f;
                    int kvOffset = position * kvStride + kvHead * headDimension;
                    for (int dimension = 0; dimension < headDimension; dimension++) {
                        dot += queries[queryOffset + dimension] * keys[kvOffset + dimension];
                    }
                    attentionScores[position - start] = dot * scale;
                }
                GGUFVectorOps.softmax(attentionScores, activeLength);

                int outputOffset = token * queryHeads * headDimension + head * headDimension;
                Arrays.fill(output, outputOffset, outputOffset + headDimension, 0f);
                for (int position = start; position <= end; position++) {
                    float weight = attentionScores[position - start];
                    int valueOffset = position * kvStride + kvHead * headDimension;
                    for (int dimension = 0; dimension < headDimension; dimension++) {
                        output[outputOffset + dimension] += weight * values[valueOffset + dimension];
                    }
                }
            }
        }
        return false;
    }
}
