package tech.kayys.aljabr.models.embeddinggemma2.attention;

/**
 * Float32 reference RMSNorm used by EmbeddingGemma 2 attention and encoder blocks.
 *
 * <p>The scale is optional: query/key norms have learned scales, while the
 * value norm is unscaled. Accumulation is performed in double precision here
 * to provide a stable reference for optimized backend kernels.</p>
 */
public final class EmbeddingGemma2RmsNorm {

    private EmbeddingGemma2RmsNorm() {
    }

    /** Normalizes [sequence][heads][headDim], returning a new tensor. */
    public static float[][][] apply(float[][][] input, float[] scale, float epsilon) {
        int sequence = requireShape(input);
        int heads = input[0].length;
        int dimensions = input[0][0].length;
        if (scale != null && scale.length != dimensions) {
            throw new IllegalArgumentException("scale length must equal the last input dimension");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        if (scale != null) requireFinite(scale, "scale");

        float[][][] output = new float[sequence][heads][dimensions];
        for (int token = 0; token < sequence; token++) {
            for (int head = 0; head < heads; head++) {
                float[] source = input[token][head];
                requireFinite(source, "input");
                double meanSquare = 0.0d;
                for (float value : source) meanSquare += (double) value * value;
                double inverseRms = 1.0d / Math.sqrt(meanSquare / dimensions + epsilon);
                for (int d = 0; d < dimensions; d++) {
                    double value = source[d] * inverseRms;
                    output[token][head][d] = (float) (scale == null ? value : value * scale[d]);
                }
            }
        }
        return output;
    }

    private static int requireShape(float[][][] input) {
        if (input == null || input.length == 0 || input[0] == null
                || input[0].length == 0 || input[0][0] == null || input[0][0].length == 0) {
            throw new IllegalArgumentException("input must be a non-empty [sequence][heads][dimension] tensor");
        }
        int heads = input[0].length;
        int dimensions = input[0][0].length;
        for (float[][] token : input) {
            if (token == null || token.length != heads) {
                throw new IllegalArgumentException("input must have a rectangular head dimension");
            }
            for (float[] head : token) {
                if (head == null || head.length != dimensions) {
                    throw new IllegalArgumentException("input must have a rectangular feature dimension");
                }
            }
        }
        return input.length;
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain only finite values");
            }
        }
    }
}
