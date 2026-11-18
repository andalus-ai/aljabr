package tech.kayys.aljabr.models.embeddinggemma2.attention;

/**
 * Checkpoint-backed EmbeddingGemma 2 gated MLP reference operation.
 *
 * <p>Computes {@code down(gelu_tanh(gate(x)) * up(x))}. All matrices are
 * output-major and bias-free, as in the reference architecture.</p>
 */
public final class EmbeddingGemma2Mlp {

    private final int hiddenSize;
    private final int intermediateSize;
    private final Weights weights;

    public EmbeddingGemma2Mlp(int hiddenSize, int intermediateSize, Weights weights) {
        if (hiddenSize <= 0 || intermediateSize <= 0) {
            throw new IllegalArgumentException("hidden and intermediate sizes must be positive");
        }
        if (weights == null) throw new IllegalArgumentException("weights must not be null");
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.weights = weights;
        requireMatrix(weights.gate(), intermediateSize, hiddenSize, "gate");
        requireMatrix(weights.up(), intermediateSize, hiddenSize, "up");
        requireMatrix(weights.down(), hiddenSize, intermediateSize, "down");
    }

    public float[][] compute(float[][] input) {
        validateInput(input);
        float[][] gate = EmbeddingGemma2LinearProjection.apply(input, weights.gate(), null);
        float[][] up = EmbeddingGemma2LinearProjection.apply(input, weights.up(), null);
        float[][] gated = new float[input.length][intermediateSize];
        for (int token = 0; token < input.length; token++) {
            for (int i = 0; i < intermediateSize; i++) {
                gated[token][i] = geluTanh(gate[token][i]) * up[token][i];
            }
        }
        return EmbeddingGemma2LinearProjection.apply(gated, weights.down(), null);
    }

    private void validateInput(float[][] input) {
        if (input == null || input.length == 0) throw new IllegalArgumentException("input must be non-empty");
        for (float[] row : input) {
            if (row == null || row.length != hiddenSize) {
                throw new IllegalArgumentException("input must have hiddenSize " + hiddenSize);
            }
        }
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " weights must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " weights must have shape [" + rows + "][" + columns + "]");
            }
        }
    }

    /** Exact tanh approximation used by the configured gelu_pytorch_tanh activation. */
    static float geluTanh(float value) {
        double x = value;
        double inner = Math.sqrt(2d / Math.PI) * (x + 0.044715d * x * x * x);
        return (float) (0.5d * x * (1d + Math.tanh(inner)));
    }

    public record Weights(float[][] gate, float[][] up, float[][] down) {
    }
}
