package tech.kayys.aljabr.models.embeddinggemma2.attention;

/**
 * Float32 reference implementation of EmbeddingGemma 2 self-attention.
 *
 * <p>Inputs are already projected, normalized, and (for Q/K) rotary-encoded.
 * The implementation performs grouped-query attention without a causal mask,
 * optionally applying the symmetric inclusive sliding-window radius used by
 * local encoder layers. It deliberately carries no KV cache.</p>
 *
 * <p>Tensor layout is [sequence][heads][headDim].</p>
 */
public final class EmbeddingGemma2BidirectionalAttention {

    private EmbeddingGemma2BidirectionalAttention() {
    }

    /** Result tensors use [sequence][queryHeads][headDim] and [queryHeads][query][key]. */
    public record Result(float[][][] output, float[][][] probabilities) {
    }

    /**
     * Computes bidirectional grouped-query attention.
     *
     * @param query [sequence][queryHeads][headDim]
     * @param key [sequence][keyValueHeads][headDim]
     * @param value [sequence][keyValueHeads][headDim]
     * @param positionIds explicit absolute position for each token
     * @param validKeys optional key-validity mask; null means every key is valid
     * @param slidingWindowRadius optional inclusive symmetric radius; null means full attention
     * @param scaling attention-score multiplier (EmbeddingGemma 2 uses 1.0)
     */
    public static Result compute(float[][][] query, float[][][] key, float[][][] value,
            int[] positionIds, boolean[] validKeys, Integer slidingWindowRadius, float scaling) {
        Shape q = shape(query, "query");
        Shape k = shape(key, "key");
        Shape v = shape(value, "value");
        if (q.sequence != k.sequence || k.sequence != v.sequence) {
            throw new IllegalArgumentException("query, key, and value sequence lengths must match");
        }
        if (k.heads != v.heads || q.dimension != k.dimension || k.dimension != v.dimension) {
            throw new IllegalArgumentException("query/key/value head dimensions are incompatible");
        }
        if (q.heads % k.heads != 0) {
            throw new IllegalArgumentException("query head count must be divisible by key/value head count");
        }
        if (positionIds == null || positionIds.length != q.sequence) {
            throw new IllegalArgumentException("positionIds length must equal sequence length");
        }
        for (int position : positionIds) {
            if (position < 0) throw new IllegalArgumentException("positionIds must be non-negative");
        }
        if (validKeys != null && validKeys.length != q.sequence) {
            throw new IllegalArgumentException("validKeys length must equal sequence length");
        }
        if (slidingWindowRadius != null && slidingWindowRadius < 0) {
            throw new IllegalArgumentException("slidingWindowRadius must be non-negative");
        }
        if (!Float.isFinite(scaling) || scaling <= 0f) {
            throw new IllegalArgumentException("scaling must be finite and positive");
        }
        requireFinite(query, "query");
        requireFinite(key, "key");
        requireFinite(value, "value");

        int groups = q.heads / k.heads;
        int dimension = q.dimension;
        float[][][] output = new float[q.sequence][q.heads][dimension];
        float[][][] probabilities = new float[q.heads][q.sequence][q.sequence];

        for (int head = 0; head < q.heads; head++) {
            int kvHead = head / groups;
            for (int token = 0; token < q.sequence; token++) {
                double maxScore = Double.NEGATIVE_INFINITY;
                double[] scores = new double[q.sequence];

                for (int source = 0; source < q.sequence; source++) {
                    if (!isVisible(source, token, positionIds, validKeys, slidingWindowRadius)) {
                        scores[source] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double dot = 0.0d;
                    for (int d = 0; d < dimension; d++) {
                        dot += (double) query[token][head][d] * key[source][kvHead][d];
                    }
                    double score = dot * scaling;
                    scores[source] = score;
                    if (score > maxScore) maxScore = score;
                }

                if (maxScore == Double.NEGATIVE_INFINITY) {
                    throw new IllegalArgumentException(
                            "attention query at sequence index " + token + " has no visible keys");
                }

                double denominator = 0.0d;
                for (int source = 0; source < q.sequence; source++) {
                    if (scores[source] != Double.NEGATIVE_INFINITY) {
                        denominator += Math.exp(scores[source] - maxScore);
                    }
                }

                for (int source = 0; source < q.sequence; source++) {
                    if (scores[source] == Double.NEGATIVE_INFINITY) continue;
                    float probability = (float) (Math.exp(scores[source] - maxScore) / denominator);
                    probabilities[head][token][source] = probability;
                    for (int d = 0; d < dimension; d++) {
                        output[token][head][d] += probability * value[source][kvHead][d];
                    }
                }
            }
        }
        return new Result(output, probabilities);
    }

    private static boolean isVisible(int keyIndex, int queryIndex, int[] positions,
            boolean[] validKeys, Integer radius) {
        if (validKeys != null && !validKeys[keyIndex]) return false;
        return radius == null || Math.abs((long) positions[keyIndex] - positions[queryIndex]) <= radius;
    }

    private static Shape shape(float[][][] tensor, String name) {
        if (tensor == null || tensor.length == 0 || tensor[0] == null
                || tensor[0].length == 0 || tensor[0][0] == null || tensor[0][0].length == 0) {
            throw new IllegalArgumentException(name + " must be a non-empty [sequence][heads][headDim] tensor");
        }
        int heads = tensor[0].length;
        int dimension = tensor[0][0].length;
        for (float[][] token : tensor) {
            if (token == null || token.length != heads) {
                throw new IllegalArgumentException(name + " must have a consistent head count");
            }
            for (float[] head : token) {
                if (head == null || head.length != dimension) {
                    throw new IllegalArgumentException(name + " must have a consistent head dimension");
                }
            }
        }
        return new Shape(tensor.length, heads, dimension);
    }

    private static void requireFinite(float[][][] tensor, String name) {
        for (float[][] token : tensor) {
            for (float[] head : token) {
                for (float value : head) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException(name + " must contain only finite values");
                    }
                }
            }
        }
    }

    private record Shape(int sequence, int heads, int dimension) {
    }
}
