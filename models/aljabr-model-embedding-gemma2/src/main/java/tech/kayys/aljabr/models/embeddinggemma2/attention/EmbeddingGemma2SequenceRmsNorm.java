package tech.kayys.aljabr.models.embeddinggemma2.attention;

/** Float32 RMSNorm reference for [sequence][hiddenSize] encoder activations. */
public final class EmbeddingGemma2SequenceRmsNorm {

    private EmbeddingGemma2SequenceRmsNorm() {
    }

    /**
     * Applies the checkpoint scale after RMS normalization. A null scale is
     * accepted only for explicitly unscaled norms.
     */
    public static float[][] apply(float[][] input, float[] scale, float epsilon) {
        int width = validate(input);
        if (scale != null && scale.length != width) {
            throw new IllegalArgumentException("scale length must equal hidden size");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        if (scale != null) requireFinite(scale, "scale");

        float[][] output = new float[input.length][width];
        for (int token = 0; token < input.length; token++) {
            requireFinite(input[token], "input");
            double meanSquare = 0d;
            for (float value : input[token]) meanSquare += (double) value * value;
            double inverseRms = 1d / Math.sqrt(meanSquare / width + epsilon);
            for (int d = 0; d < width; d++) {
                double normalized = input[token][d] * inverseRms;
                output[token][d] = (float) (scale == null ? normalized : normalized * scale[d]);
            }
        }
        return output;
    }

    private static int validate(float[][] input) {
        if (input == null || input.length == 0 || input[0] == null || input[0].length == 0) {
            throw new IllegalArgumentException("input must be a non-empty matrix");
        }
        int width = input[0].length;
        for (float[] row : input) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException("input must be rectangular");
            }
        }
        return width;
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain only finite values");
            }
        }
    }
}
