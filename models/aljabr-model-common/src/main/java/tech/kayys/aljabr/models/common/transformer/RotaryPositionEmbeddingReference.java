package tech.kayys.aljabr.models.common.transformer;

/**
 * Backend-neutral float32 reference rotary position embedding (RoPE).
 *
 * <p>Input layout is {@code [sequence][heads][headDimension]}. Uses the
 * split-half rotation convention {@code rotateHalf([x1,x2])=[-x2,x1]}.
 * Query and key projections may use this operation; values must not.</p>
 */
public final class RotaryPositionEmbeddingReference {
    private RotaryPositionEmbeddingReference() {
    }

    public static float[][][] apply(float[][][] input, int[] positions,
            double theta, Double partialRotaryFactor) {
        Shape shape = requireShape(input);
        if (positions == null || positions.length != shape.sequence) {
            throw new IllegalArgumentException("positions length must equal sequence length");
        }
        if (!Double.isFinite(theta) || theta <= 0d) {
            throw new IllegalArgumentException("theta must be finite and positive");
        }
        double factor = partialRotaryFactor == null ? 1d : partialRotaryFactor;
        if (!Double.isFinite(factor) || factor <= 0d || factor > 1d) {
            throw new IllegalArgumentException("partialRotaryFactor must be in (0, 1]");
        }
        int rotaryDimension = ((int) Math.floor(shape.dimension * factor)) & ~1;
        if (rotaryDimension == 0) {
            throw new IllegalArgumentException("rotary dimension must be at least two");
        }

        float[][][] output = new float[shape.sequence][shape.heads][shape.dimension];
        int half = rotaryDimension / 2;
        for (int token = 0; token < shape.sequence; token++) {
            for (int head = 0; head < shape.heads; head++) {
                float[] source = input[token][head];
                for (int dimension = 0; dimension < shape.dimension; dimension++) {
                    if (!Float.isFinite(source[dimension])) {
                        throw new IllegalArgumentException("input must contain only finite values");
                    }
                    output[token][head][dimension] = source[dimension];
                }
                for (int dimension = 0; dimension < half; dimension++) {
                    double inverseFrequency = 1d / Math.pow(theta, (2d * dimension) / rotaryDimension);
                    double angle = positions[token] * inverseFrequency;
                    double cosine = Math.cos(angle);
                    double sine = Math.sin(angle);
                    double first = source[dimension];
                    double second = source[dimension + half];
                    output[token][head][dimension] = (float) (first * cosine - second * sine);
                    output[token][head][dimension + half] = (float) (second * cosine + first * sine);
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

    private record Shape(int sequence, int heads, int dimension) {
    }
}
