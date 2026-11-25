package tech.kayys.aljabr.backend.cpu;

import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheAttentionBlockConsumer;
import tech.kayys.aljabr.cache.KVCacheSession;

/**
 * CPU reference attention over a session-backed KV cache.
 *
 * <p>The implementation consumes the cache's native block-read capability and
 * maintains online-softmax state across callbacks. It therefore does not
 * materialize the requested K/V sequence or a score/probability vector.</p>
 *
 * <p>Query heads are mapped to KV heads using the standard contiguous GQA
 * grouping rule. Runtime scheduling and cache lifecycle remain outside this
 * class; the session enforces position visibility.</p>
 */
public final class CpuKVCacheStreamingAttention {

    private CpuKVCacheStreamingAttention() {
    }

    /**
     * Computes one query head against a visible cache range.
     *
     * @param session sequence-scoped cache session
     * @param layer cache layer
     * @param query one query vector of {@code headDimension} elements
     * @param queryHeadIndex index in the query-head group
     * @param queryHeadCount total query heads for this layer
     * @param queryPosition absolute query position, used by the causal mask
     * @param firstKeyPosition first absolute key position to read
     * @param tokenCount number of key positions to read
     * @param causal whether keys after the query position are excluded
     * @return newly allocated attention output vector
     */
    public static float[] compute(
            KVCacheSession session,
            int layer,
            float[] query,
            int queryHeadIndex,
            int queryHeadCount,
            int queryPosition,
            int firstKeyPosition,
            int tokenCount,
            boolean causal) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(query, "query");

        int kvHeads = session.layout().kvHeads(layer);
        int headDimension = session.layout().headDimension(layer);
        if (query.length != headDimension) {
            throw new IllegalArgumentException("query length must match the cache head dimension");
        }
        if (queryHeadCount <= 0 || queryHeadIndex < 0 || queryHeadIndex >= queryHeadCount
                || queryHeadCount % kvHeads != 0) {
            throw new IllegalArgumentException(
                    "query head count must be positive and divisible by the KV head count");
        }
        if (queryPosition < 0 || firstKeyPosition < 0 || tokenCount <= 0
                || firstKeyPosition > Integer.MAX_VALUE - tokenCount) {
            throw new IllegalArgumentException("invalid query or key range");
        }

        int queriesPerKvHead = queryHeadCount / kvHeads;
        int kvHead = queryHeadIndex / queriesPerKvHead;
        int stride = session.layout().layerStride(layer);
        int headOffset = Math.multiplyExact(kvHead, headDimension);
        double scale = 1.0d / Math.sqrt(headDimension);

        double[] weightedValues = new double[headDimension];
        double[] running = {Double.NEGATIVE_INFINITY, 0.0d};
        KVCacheAttentionBlockConsumer consumer =
                (blockFirst, blockCount, keys, values, elementOffset) -> {
                    for (int token = 0; token < blockCount; token++) {
                        int position = blockFirst + token;
                        if (causal && position > queryPosition) {
                            continue;
                        }
                        int vectorOffset = Math.addExact(
                                elementOffset,
                                Math.addExact(Math.multiplyExact(token, stride), headOffset));
                        double score = 0.0d;
                        for (int d = 0; d < headDimension; d++) {
                            score += (double) query[d] * keys[vectorOffset + d];
                        }
                        score *= scale;

                        double nextMax = Math.max(running[0], score);
                        double previousScale = running[1] == 0.0d
                                ? 0.0d : Math.exp(running[0] - nextMax);
                        double currentScale = Math.exp(score - nextMax);
                        double nextDenominator = running[1] * previousScale + currentScale;
                        for (int d = 0; d < headDimension; d++) {
                            weightedValues[d] = weightedValues[d] * previousScale
                                    + currentScale * values[vectorOffset + d];
                        }
                        running[0] = nextMax;
                        running[1] = nextDenominator;
                    }
                };

        session.readBlocksForAttention(layer, firstKeyPosition, tokenCount, consumer);
        float[] output = new float[headDimension];
        if (running[1] != 0.0d) {
            for (int d = 0; d < headDimension; d++) {
                output[d] = (float) (weightedValues[d] / running[1]);
            }
        }
        return output;
    }
}
