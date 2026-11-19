package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Float32 axial two-dimensional RoPE used by the Gemma4 vision encoder.
 *
 * <p>The head is split into two equal spatial-axis regions. Each region uses
 * the same inverse-frequency schedule, with the first coordinate applied to
 * the first region and the second coordinate to the second region. Within
 * each region, frequencies are duplicated to match the reference H-H-W-W
 * recomposition before the standard rotate-half operation.</p>
 *
 * <p>Input coordinates are {@code (x,y)}. A {@code (-1,-1)} padding
 * coordinate is accepted and transformed deterministically; callers must
 * still exclude padded patches from attention using the validity mask.</p>
 */
public final class EmbeddingGemma2VisionRotaryEmbedding {
    private final int headDimension;
    private final int axisDimension;
    private final double[] inverseFrequencies;

    public EmbeddingGemma2VisionRotaryEmbedding(int headDimension, double theta) {
        if (headDimension < 4 || headDimension % 4 != 0) {
            throw new IllegalArgumentException("headDimension must be >= 4 and divisible by 4");
        }
        if (!Double.isFinite(theta) || theta <= 0d) {
            throw new IllegalArgumentException("theta must be finite and positive");
        }
        this.headDimension = headDimension;
        this.axisDimension = headDimension / 2;
        int frequencyCount = axisDimension / 2;
        this.inverseFrequencies = new double[frequencyCount];
        for (int i = 0; i < frequencyCount; i++) {
            int exponentNumerator = i * 2;
            inverseFrequencies[i] = 1d / Math.pow(theta, (double) exponentNumerator / axisDimension);
        }
    }

    public int headDimension() {
        return headDimension;
    }

    /** Returns cosine and sine tables in [sequence][headDimension] layout. */
    public Embeddings compute(int[][] pixelPositionIds) {
        validatePositions(pixelPositionIds);
        float[][] cosine = new float[pixelPositionIds.length][headDimension];
        float[][] sine = new float[pixelPositionIds.length][headDimension];
        for (int token = 0; token < pixelPositionIds.length; token++) {
            for (int axis = 0; axis < 2; axis++) {
                int coordinate = pixelPositionIds[token][axis];
                int offset = axis * axisDimension;
                for (int i = 0; i < inverseFrequencies.length; i++) {
                    double angle = coordinate * inverseFrequencies[i];
                    float cos = (float) Math.cos(angle);
                    float sin = (float) Math.sin(angle);
                    cosine[token][offset + i] = cos;
                    cosine[token][offset + i + inverseFrequencies.length] = cos;
                    sine[token][offset + i] = sin;
                    sine[token][offset + i + inverseFrequencies.length] = sin;
                }
            }
        }
        return new Embeddings(cosine, sine);
    }

    /**
     * Applies axial RoPE to query or key vectors in [sequence][heads][headDimension]
     * layout. The input is never mutated.
     */
    public float[][][] apply(float[][][] states, int[][] pixelPositionIds) {
        Objects.requireNonNull(states, "states must not be null");
        Embeddings embeddings = compute(pixelPositionIds);
        if (states.length != pixelPositionIds.length || states.length == 0) {
            throw new IllegalArgumentException("states and positions must have the same non-zero sequence length");
        }
        if (states[0] == null || states[0].length == 0) {
            throw new IllegalArgumentException("states must contain at least one attention head");
        }
        int heads = states[0].length;
        float[][][] output = new float[states.length][heads][headDimension];
        for (int token = 0; token < states.length; token++) {
            if (states[token] == null || states[token].length != heads) {
                throw new IllegalArgumentException("states must have a rectangular [sequence][heads][headDimension] shape");
            }
            for (int head = 0; head < heads; head++) {
                float[] vector = states[token][head];
                if (vector == null || vector.length != headDimension) {
                    throw new IllegalArgumentException("each attention head must have dimension " + headDimension);
                }
                for (float value : vector) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("states must contain only finite values");
                    }
                }
                applyAxis(vector, output[token][head], embeddings.cosine()[token],
                        embeddings.sine()[token], 0);
                applyAxis(vector, output[token][head], embeddings.cosine()[token],
                        embeddings.sine()[token], axisDimension);
            }
        }
        return output;
    }

    private void applyAxis(float[] input, float[] output, float[] cosine, float[] sine, int offset) {
        int half = axisDimension / 2;
        for (int i = 0; i < half; i++) {
            float first = input[offset + i];
            float second = input[offset + i + half];
            float cos = cosine[offset + i];
            float sin = sine[offset + i];
            output[offset + i] = first * cos - second * sin;
            output[offset + i + half] = second * cos + first * sin;
        }
    }

    private static void validatePositions(int[][] positions) {
        if (positions == null || positions.length == 0) {
            throw new IllegalArgumentException("pixelPositionIds must be non-empty");
        }
        for (int i = 0; i < positions.length; i++) {
            if (positions[i] == null || positions[i].length != 2) {
                throw new IllegalArgumentException("pixel position " + i + " must contain (x,y)");
            }
            int x = positions[i][0], y = positions[i][1];
            if (!((x == -1 && y == -1) || (x >= 0 && y >= 0))) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
        }
    }

    public record Embeddings(float[][] cosine, float[][] sine) {
        public Embeddings {
            Objects.requireNonNull(cosine, "cosine must not be null");
            Objects.requireNonNull(sine, "sine must not be null");
            if (cosine.length != sine.length) {
                throw new IllegalArgumentException("cosine and sine sequence lengths must match");
            }
        }
    }
}
