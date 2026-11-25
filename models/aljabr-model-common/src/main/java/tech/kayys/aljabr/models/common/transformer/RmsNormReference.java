package tech.kayys.aljabr.models.common.transformer;

/**
 * Backend-neutral float32 reference RMS normalization for transformer layers.
 *
 * <p>Input layout is {@code [sequence][heads][headDimension]}. The operation
 * normalizes the last dimension independently for each token and head, then
 * applies an optional learned scale. It intentionally does not subtract the
 * mean, matching T5-style RMS normalization.</p>
 */
public final class RmsNormReference {
    private RmsNormReference() {
    }

    public static float[][][] apply(float[][][] input, float[] scale, float epsilon) {
        Shape shape = requireShape(input);
        if (scale != null && scale.length != shape.dimension) {
            throw new IllegalArgumentException("scale length must equal the last input dimension");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        if (scale != null) requireFinite(scale, "scale");

        float[][][] output = new float[shape.sequence][shape.heads][shape.dimension];
        for (int token = 0; token < shape.sequence; token++) {
            for (int head = 0; head < shape.heads; head++) {
                float[] source = input[token][head];
                requireFinite(source, "input");
                double meanSquare = 0d;
                for (float value : source) meanSquare += (double) value * value;
                double inverseRms = 1d / Math.sqrt(meanSquare / shape.dimension + epsilon);
                for (int dimension = 0; dimension < shape.dimension; dimension++) {
                    double normalized = source[dimension] * inverseRms;
                    output[token][head][dimension] =
                            (float) (scale == null ? normalized : normalized * scale[dimension]);
                }
            }
        }
        return output;
    }

    private static Shape requireShape(float[][][] input) {
        if (input == null || input.length == 0 || input[0] == null
                || input[0].length == 0 || input[0][0] == null || input[0][0].length == 0) {
            throw new IllegalArgumentException("input must be a non-empty [sequence][heads][dimension] tensor");
        }
        int heads = input[0].length;
        int dimension = input[0][0].length;
        for (float[][] token : input) {
            if (token == null || token.length != heads) {
                throw new IllegalArgumentException("input must have a rectangular head dimension");
            }
            for (float[] head : token) {
                if (head == null || head.length != dimension) {
                    throw new IllegalArgumentException("input must have a rectangular feature dimension");
                }
            }
        }
        return new Shape(input.length, heads, dimension);
    }

    private static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(name + " must contain only finite values");
            }
        }
    }

    private record Shape(int sequence, int heads, int dimension) {
    }
}
