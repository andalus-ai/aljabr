package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2SequenceRmsNorm;

/**
 * One Gemma 4 vision encoder layer used by EmbeddingGemma 2.
 *
 * <p>Implements the reference residual order:
 * input RMSNorm -> bidirectional attention -> post-attention RMSNorm -> residual;
 * pre-feedforward RMSNorm -> gated GELU MLP -> post-feedforward RMSNorm -> residual.
 * All four norm scales and all MLP matrices are checkpoint supplied.</p>
 *
 * <p>The attention operation is injected so its axial 2D RoPE and attention
 * implementation remain a separately testable component. It must implement
 * non-causal attention and honor {@code validPatches} as a key mask.</p>
 */
public final class EmbeddingGemma2VisionEncoderLayer {
    private final int hiddenSize;
    private final float epsilon;
    private final Attention attention;
    private final EmbeddingGemma2Mlp mlp;
    private final Weights weights;

    public EmbeddingGemma2VisionEncoderLayer(
            int hiddenSize,
            int intermediateSize,
            float epsilon,
            Attention attention,
            EmbeddingGemma2Mlp.Weights mlpWeights,
            Weights weights) {
        if (hiddenSize <= 0 || intermediateSize <= 0) {
            throw new IllegalArgumentException("hidden and intermediate sizes must be positive");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        this.hiddenSize = hiddenSize;
        this.epsilon = epsilon;
        this.attention = Objects.requireNonNull(attention, "attention must not be null");
        this.mlp = new EmbeddingGemma2Mlp(hiddenSize, intermediateSize,
                Objects.requireNonNull(mlpWeights, "mlpWeights must not be null"));
        this.weights = Objects.requireNonNull(weights, "weights must not be null");
        requireVector(weights.inputNormScale(), hiddenSize, "inputNormScale");
        requireVector(weights.postAttentionNormScale(), hiddenSize, "postAttentionNormScale");
        requireVector(weights.preFeedforwardNormScale(), hiddenSize, "preFeedforwardNormScale");
        requireVector(weights.postFeedforwardNormScale(), hiddenSize, "postFeedforwardNormScale");
    }

    /**
     * Computes one image's encoder sequence.
     *
     * @param hiddenStates patch states [sequence][hiddenSize]
     * @param pixelPositionIds (x,y) coordinates [sequence][2], padding is (-1,-1)
     * @param validPatches optional true=valid patch mask
     */
    public float[][] compute(float[][] hiddenStates, int[][] pixelPositionIds, boolean[] validPatches) {
        validateStates(hiddenStates);
        validatePositions(pixelPositionIds, hiddenStates.length);
        if (validPatches != null && validPatches.length != hiddenStates.length) {
            throw new IllegalArgumentException("validPatches length must equal sequence length");
        }

        float[][] residual = copy(hiddenStates);
        float[][] normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                hiddenStates, weights.inputNormScale(), epsilon);
        float[][] attended = attention.compute(normalized, pixelPositionIds, validPatches);
        validateStates(attended);
        if (attended.length != hiddenStates.length) {
            throw new IllegalArgumentException("attention output sequence length must not change");
        }
        attended = EmbeddingGemma2SequenceRmsNorm.apply(
                attended, weights.postAttentionNormScale(), epsilon);
        float[][] afterAttention = add(residual, attended);

        residual = copy(afterAttention);
        normalized = EmbeddingGemma2SequenceRmsNorm.apply(
                afterAttention, weights.preFeedforwardNormScale(), epsilon);
        float[][] feedforward = mlp.compute(normalized);
        feedforward = EmbeddingGemma2SequenceRmsNorm.apply(
                feedforward, weights.postFeedforwardNormScale(), epsilon);
        return add(residual, feedforward);
    }

    private void validateStates(float[][] states) {
        if (states == null || states.length == 0) {
            throw new IllegalArgumentException("hidden states must be non-empty");
        }
        for (float[] row : states) {
            if (row == null || row.length != hiddenSize) {
                throw new IllegalArgumentException("hidden states must have shape [sequence][" + hiddenSize + "]");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) throw new IllegalArgumentException("hidden states must be finite");
            }
        }
    }

    private static void validatePositions(int[][] positions, int sequence) {
        if (positions == null || positions.length != sequence) {
            throw new IllegalArgumentException("pixelPositionIds length must equal sequence length");
        }
        for (int i = 0; i < positions.length; i++) {
            if (positions[i] == null || positions[i].length != 2) {
                throw new IllegalArgumentException("pixel position " + i + " must contain (x,y)");
            }
            int x = positions[i][0], y = positions[i][1];
            if (!((x == -1 && y == -1) || (x >= 0 && y >= 0))) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
            if (x == -1 && y == -1 && positions.length == 1) {
                throw new IllegalArgumentException("an image must contain at least one valid patch");
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must be finite");
        }
    }

    private static float[][] add(float[][] a, float[][] b) {
        float[][] out = new float[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) out[i][j] = a[i][j] + b[i][j];
        }
        return out;
    }

    private static float[][] copy(float[][] input) {
        float[][] out = new float[input.length][];
        for (int i = 0; i < input.length; i++) out[i] = input[i].clone();
        return out;
    }

    @FunctionalInterface
    public interface Attention {
        float[][] compute(float[][] normalizedStates, int[][] pixelPositionIds, boolean[] validPatches);
    }

    /** Trainable scales for the four RMSNorm operations in this encoder layer. */
    public record Weights(
            float[] inputNormScale,
            float[] postAttentionNormScale,
            float[] preFeedforwardNormScale,
            float[] postFeedforwardNormScale) {
    }
}
