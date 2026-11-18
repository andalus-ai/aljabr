package tech.kayys.aljabr.models.embeddinggemma2.output;

/**
 * Float32 reference implementation of the learned 512-to-768 embedding head.
 *
 * <p>Weights must come from the model checkpoint. This class deliberately does
 * not initialize or invent model parameters. The matrix is output-major:
 * {@code weights[outputDimension][inputDimension]}.
 */
public final class EmbeddingGemma2Projection {

    public static final int INPUT_DIMENSION = 512;
    public static final int OUTPUT_DIMENSION = 768;

    private EmbeddingGemma2Projection() {
    }

    /**
     * Applies the checkpoint's affine projection to one pooled hidden vector.
     *
     * @param input pooled 512-dimensional hidden state
     * @param weights checkpoint projection weights [768][512]
     * @param bias checkpoint projection bias [768], or null for a bias-free head
     * @return projected 768-dimensional vector
     */
    public static float[] project(float[] input, float[][] weights, float[] bias) {
        if (input == null || input.length != INPUT_DIMENSION) {
            throw new IllegalArgumentException("input must have dimension " + INPUT_DIMENSION);
        }
        if (weights == null || weights.length != OUTPUT_DIMENSION) {
            throw new IllegalArgumentException("weights must have shape [768][512]");
        }
        if (bias != null && bias.length != OUTPUT_DIMENSION) {
            throw new IllegalArgumentException("bias must have dimension " + OUTPUT_DIMENSION);
        }

        for (float value : input) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("input must contain only finite values");
            }
        }
        for (int output = 0; output < OUTPUT_DIMENSION; output++) {
            if (weights[output] == null || weights[output].length != INPUT_DIMENSION) {
                throw new IllegalArgumentException("weights must have shape [768][512]");
            }
            if (bias != null && !Float.isFinite(bias[output])) {
                throw new IllegalArgumentException("bias must contain only finite values");
            }
            for (float weight : weights[output]) {
                if (!Float.isFinite(weight)) {
                    throw new IllegalArgumentException("weights must contain only finite values");
                }
            }
        }

        float[] projected = new float[OUTPUT_DIMENSION];
        for (int output = 0; output < OUTPUT_DIMENSION; output++) {
            double sum = bias == null ? 0.0d : bias[output];
            for (int dimension = 0; dimension < INPUT_DIMENSION; dimension++) {
                sum += (double) weights[output][dimension] * input[dimension];
            }
            projected[output] = (float) sum;
        }
        return projected;
    }
}
