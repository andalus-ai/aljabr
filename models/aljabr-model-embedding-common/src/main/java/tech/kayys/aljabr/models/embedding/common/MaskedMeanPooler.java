package tech.kayys.aljabr.models.embedding.common;

/**
 * Computes attention-mask-aware mean pooling over sequence-major token states.
 *
 * <p>Masked padding positions do not contribute to either the sum or divisor.
 * The input arrays are never modified. Accumulation uses double precision
 * before returning float32 values.</p>
 */
public final class MaskedMeanPooler {
    private MaskedMeanPooler() {
    }

    /**
     * @param tokenStates token states in {@code [sequenceLength][hiddenSize]} layout
     * @param attentionMask {@code true} for positions included in the mean
     * @return a new pooled vector
     */
    public static float[] meanPool(float[][] tokenStates, boolean[] attentionMask) {
        if (tokenStates == null || tokenStates.length == 0) {
            throw new IllegalArgumentException("tokenStates must contain at least one token");
        }
        if (attentionMask == null || attentionMask.length != tokenStates.length) {
            throw new IllegalArgumentException("attentionMask length must match token count");
        }
        int hiddenSize = -1;
        for (int token = 0; token < tokenStates.length; token++) {
            if (!attentionMask[token]) {
                continue;
            }
            float[] state = tokenStates[token];
            if (state == null || state.length == 0) {
                throw new IllegalArgumentException("unmasked token states must have a non-zero hidden dimension");
            }
            if (hiddenSize < 0) {
                hiddenSize = state.length;
            } else if (state.length != hiddenSize) {
                throw new IllegalArgumentException("all unmasked token states must have the same hidden dimension");
            }
        }
        if (hiddenSize < 0) {
            throw new IllegalArgumentException("attentionMask must include at least one token");
        }

        double[] sums = new double[hiddenSize];
        int includedTokens = 0;
        for (int token = 0; token < tokenStates.length; token++) {
            if (!attentionMask[token]) {
                continue;
            }
            float[] state = tokenStates[token];
            for (int dimension = 0; dimension < hiddenSize; dimension++) {
                float value = state[dimension];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("unmasked token states must contain only finite values");
                }
                sums[dimension] += value;
            }
            includedTokens++;
        }

        float[] pooled = new float[hiddenSize];
        for (int dimension = 0; dimension < hiddenSize; dimension++) {
            pooled[dimension] = (float) (sums[dimension] / includedTokens);
        }
        return pooled;
    }
}
