package tech.kayys.aljabr.backend.hat;

import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

import static hat.NDRange.of1D;

/**
 * Stable online-softmax decode attention over sequence-major GQA cache arrays.
 *
 * <p>Each work item computes one output element. The online recurrence avoids
 * a temporary score array and keeps the kernel independent of sequence length
 * allocation. This is a correctness-first staged path, not a tiled/high-
 * throughput attention kernel.</p>
 */
public final class HatKVCacheAttentionKernel {
    private HatKVCacheAttentionKernel() {
    }

    @Reflect
    public static void decode(
            KernelContext kc,
            F32Array query,
            F32Array keys,
            F32Array values,
            F32Array output,
            int queryHeads,
            int kvHeads,
            int headDimension,
            int sequenceLength,
            int lastPosition,
            float scale) {
        int outputIndex = kc.gix;
        int outputElements = queryHeads * headDimension;
        if (outputIndex >= outputElements) {
            return;
        }

        int queryHead = outputIndex / headDimension;
        int outputDimension = outputIndex % headDimension;
        int kvHead = queryHead / (queryHeads / kvHeads);
        int queryOffset = queryHead * headDimension;
        int kvStride = kvHeads * headDimension;

        float maxScore = -Float.MAX_VALUE;
        float denominator = 0.0f;
        float weightedValue = 0.0f;

        for (int position = 0; position <= lastPosition; position++) {
            int keyOffset = position * kvStride + kvHead * headDimension;
            float score = 0.0f;
            for (int dimension = 0; dimension < headDimension; dimension++) {
                score += query.array(queryOffset + dimension)
                        * keys.array(keyOffset + dimension);
            }
            score *= scale;

            int valueOffset = keyOffset + outputDimension;
            float value = values.array(valueOffset);
            if (score > maxScore) {
                float rescale = (float) Math.exp(maxScore - score);
                weightedValue = weightedValue * rescale + value;
                denominator = denominator * rescale + 1.0f;
                maxScore = score;
            } else {
                float weight = (float) Math.exp(score - maxScore);
                weightedValue += weight * value;
                denominator += weight;
            }
        }

        output.array(outputIndex, weightedValue / denominator);
    }

    @Reflect
    public static void dispatch(
            ComputeContext cc,
            F32Array query,
            F32Array keys,
            F32Array values,
            F32Array output,
            int queryHeads,
            int kvHeads,
            int headDimension,
            int sequenceLength,
            int lastPosition,
            float scale) {
        cc.dispatchKernel(of1D(queryHeads * headDimension),
                kc -> decode(kc, query, keys, values, output, queryHeads,
                        kvHeads, headDimension, sequenceLength, lastPosition, scale));
    }
}
