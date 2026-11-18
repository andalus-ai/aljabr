package tech.kayys.aljabr.models.embeddinggemma2.attention;

/**
 * Checkpoint-backed float32 linear projection.
 *
 * <p>Weights use output-major {@code [output][input]} layout, matching the
 * embedding projection reference operations. No parameter initialization is
 * performed: callers must supply checkpoint tensors.</p>
 */
public final class EmbeddingGemma2LinearProjection {

    private EmbeddingGemma2LinearProjection() {
    }

    /** Applies {@code y = W x + b} to a [sequence][input] tensor. */
    public static float[][] apply(float[][] input, float[][] weights, float[] bias) {
        int inputSize = validateInput(input);
        int outputSize = validateWeights(weights, inputSize);
        if (bias != null && bias.length != outputSize) {
            throw new IllegalArgumentException("bias length must equal projection output size");
        }
        if (bias != null) requireFinite(bias, "bias");

        float[][] output = new float[input.length][outputSize];
        for (int token = 0; token < input.length; token++) {
            requireFinite(input[token], "input");
            for (int out = 0; out < outputSize; out++) {
                double sum = bias == null ? 0.0d : bias[out];
                for (int in = 0; in < inputSize; in++) {
                    sum += (double) weights[out][in] * input[token][in];
                }
                output[token][out] = (float) sum;
            }
        }
        return output;
    }

    private static int validateInput(float[][] input) {
        if (input == null || input.length == 0 || input[0] == null || input[0].length == 0) {
            throw new IllegalArgumentException("input must be a non-empty [sequence][features] tensor");
        }
        int width = input[0].length;
        for (float[] row : input) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException("input must be rectangular");
            }
        }
        return width;
    }

    private static int validateWeights(float[][] weights, int inputSize) {
        if (weights == null || weights.length == 0 || weights[0] == null
                || weights[0].length != inputSize) {
            throw new IllegalArgumentException("weights must have shape [output][" + inputSize + "]");
        }
        for (float[] row : weights) {
            if (row == null || row.length != inputSize) {
                throw new IllegalArgumentException("weights must be rectangular with the input feature width");
            }
            requireFinite(row, "weights");
        }
        return weights.length;
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain only finite values");
            }
        }
    }
}
