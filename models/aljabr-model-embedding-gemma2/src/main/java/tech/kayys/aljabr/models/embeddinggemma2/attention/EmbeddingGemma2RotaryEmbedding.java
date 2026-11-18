package tech.kayys.aljabr.models.embeddinggemma2.attention;

/**
 * Float32 reference RoPE for EmbeddingGemma 2's per-layer attention geometry.
 *
 * <p>Uses the split-half convention from the Hugging Face reference:
 * rotate_half([x1, x2]) = [-x2, x1], with duplicated inverse frequencies.
 * Query and key are rotated; values are never passed to this operation.</p>
 */
public final class EmbeddingGemma2RotaryEmbedding {

    private EmbeddingGemma2RotaryEmbedding() {
    }

    /**
     * Applies RoPE to [sequence][heads][headDim] using explicit position ids.
     *
     * @param input query or key tensor
     * @param positions one position id per sequence token
     * @param theta RoPE base for this layer type
     * @param partialRotaryFactor fraction of each head dimension to rotate (null means 1)
     * @return a new tensor; input is not mutated
     */
    public static float[][][] apply(float[][][] input, int[] positions,
            double theta, Double partialRotaryFactor) {
        int sequence = requireShape(input);
        int heads = input[0].length;
        int headDim = input[0][0].length;
        if (positions == null || positions.length != sequence) {
            throw new IllegalArgumentException("positions length must equal sequence length");
        }
        if (!Double.isFinite(theta) || theta <= 0.0d) {
            throw new IllegalArgumentException("theta must be finite and positive");
        }
        double factor = partialRotaryFactor == null ? 1.0d : partialRotaryFactor;
        if (!Double.isFinite(factor) || factor <= 0.0d || factor > 1.0d) {
            throw new IllegalArgumentException("partialRotaryFactor must be in (0, 1]");
        }
        int rotaryDim = ((int) Math.floor(headDim * factor)) & ~1;
        if (rotaryDim == 0) throw new IllegalArgumentException("rotary dimension must be at least two");
        for (int position : positions) {
            if (position < 0) throw new IllegalArgumentException("position ids must not be negative");
        }

        float[][][] output = new float[sequence][heads][headDim];
        int half = rotaryDim / 2;
        for (int token = 0; token < sequence; token++) {
            for (int head = 0; head < heads; head++) {
                float[] source = input[token][head];
                for (int d = 0; d < headDim; d++) {
                    if (!Float.isFinite(source[d])) {
                        throw new IllegalArgumentException("input must contain only finite values");
                    }
                    output[token][head][d] = source[d];
                }
                for (int d = 0; d < half; d++) {
                    double inverseFrequency = 1.0d / Math.pow(theta, (2.0d * d) / rotaryDim);
                    double angle = positions[token] * inverseFrequency;
                    double cos = Math.cos(angle);
                    double sin = Math.sin(angle);
                    double first = source[d];
                    double second = source[d + half];
                    output[token][head][d] = (float) (first * cos - second * sin);
                    output[token][head][d + half] = (float) (second * cos + first * sin);
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
}
