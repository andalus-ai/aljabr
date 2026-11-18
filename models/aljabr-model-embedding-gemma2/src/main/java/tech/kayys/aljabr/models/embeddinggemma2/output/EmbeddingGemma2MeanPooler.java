package tech.kayys.aljabr.models.embeddinggemma2.output;

/**
 * Float32 reference implementation of EmbeddingGemma 2 mask-aware mean pooling.
 *
 * <p>Only positions whose attention-mask value is true contribute to the mean.
 * The input is not modified. Masked positions are intentionally not inspected,
 * because padding states are semantically excluded from the pooled result.
 */
public final class EmbeddingGemma2MeanPooler {

    private EmbeddingGemma2MeanPooler() {
    }

    /**
     * Pools token states over the unmasked sequence positions.
     *
     * @param tokenStates sequence-major token states [sequenceLength][hiddenSize]
     * @param attentionMask true for a real token, false for padding
     * @return pooled hidden state in float32
     */
    public static float[] meanPool(float[][] tokenStates, boolean[] attentionMask) {
        if (tokenStates == null || tokenStates.length == 0) {
            throw new IllegalArgumentException("tokenStates must contain at least one token");
        }
        if (attentionMask == null || attentionMask.length != tokenStates.length) {
            throw new IllegalArgumentException("attentionMask length must match token count");
        }
        if (tokenStates[0] == null || tokenStates[0].length == 0) {
            throw new IllegalArgumentException("token states must have a non-zero hidden dimension");
        }

        int hiddenSize = tokenStates[0].length;
        double[] sums = new double[hiddenSize];
        int includedTokens = 0;

        for (int token = 0; token < tokenStates.length; token++) {
            if (!attentionMask[token]) {
                continue;
            }
            float[] state = tokenStates[token];
            if (state == null || state.length != hiddenSize) {
                throw new IllegalArgumentException("all token states must have the same hidden dimension");
            }
            for (int dimension = 0; dimension < hiddenSize; dimension++) {
                float value = state[dimension];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("unmasked token states must contain only finite values");
                }
                sums[dimension] += value;
            }
            includedTokens++;
        }

        if (includedTokens == 0) {
            throw new IllegalArgumentException("attentionMask must include at least one token");
        }

        float[] pooled = new float[hiddenSize];
        for (int dimension = 0; dimension < hiddenSize; dimension++) {
            pooled[dimension] = (float) (sums[dimension] / includedTokens);
        }
        return pooled;
    }
}
