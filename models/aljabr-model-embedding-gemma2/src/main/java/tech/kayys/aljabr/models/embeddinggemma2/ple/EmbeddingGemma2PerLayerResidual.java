package tech.kayys.aljabr.models.embeddinggemma2.ple;

/**
 * Float32 reference for the third residual sub-block in an EmbeddingGemma 2
 * encoder layer, which injects that layer's projected PLE signal.
 */
public final class EmbeddingGemma2PerLayerResidual {

    private EmbeddingGemma2PerLayerResidual() {
    }

    /**
     * Computes residual + RMSNorm(W_up(GELU(W_gate(hidden)) * perLayerInput)).
     *
     * <p>All checkpoint matrices are bias-free and stored output-major, matching
     * the standard linear weight layout [out_features][in_features].
     *
     * @param hiddenStates [sequence][hiddenSize]
     * @param perLayerInput [sequence][pleSize]
     * @param gateWeights [pleSize][hiddenSize]
     * @param projectionWeights [hiddenSize][pleSize]
     * @param normScale [hiddenSize]
     * @param epsilon RMSNorm epsilon
     * @return new residual stream [sequence][hiddenSize]
     */
    public static float[][] apply(float[][] hiddenStates, float[][] perLayerInput,
            float[][] gateWeights, float[][] projectionWeights, float[] normScale, float epsilon) {
        int sequence = requireMatrix(hiddenStates, "hiddenStates");
        int hidden = hiddenStates[0].length;
        int ple = requireMatrix(perLayerInput, "perLayerInput");
        if (perLayerInput.length != sequence) {
            throw new IllegalArgumentException("perLayerInput sequence length must match hiddenStates");
        }
        requireShape(gateWeights, ple, hidden, "gateWeights");
        requireShape(projectionWeights, hidden, ple, "projectionWeights");
        if (normScale == null || normScale.length != hidden) {
            throw new IllegalArgumentException("normScale must have dimension hiddenSize");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        EmbeddingGemma2PerLayerEmbeddingProjector.requireFinite(normScale, "normScale");
        for (float[] row : hiddenStates) EmbeddingGemma2PerLayerEmbeddingProjector.requireFinite(row, "hiddenStates");
        for (float[] row : perLayerInput) EmbeddingGemma2PerLayerEmbeddingProjector.requireFinite(row, "perLayerInput");
        for (float[] row : gateWeights) EmbeddingGemma2PerLayerEmbeddingProjector.requireFinite(row, "gateWeights");
        for (float[] row : projectionWeights) EmbeddingGemma2PerLayerEmbeddingProjector.requireFinite(row, "projectionWeights");

        float[][] output = new float[sequence][hidden];
        float[] gated = new float[ple];
        for (int token = 0; token < sequence; token++) {
            for (int p = 0; p < ple; p++) {
                double sum = 0.0d;
                for (int h = 0; h < hidden; h++) sum += (double) gateWeights[p][h] * hiddenStates[token][h];
                gated[p] = geluTanh((float) sum) * perLayerInput[token][p];
            }
            float[] update = new float[hidden];
            for (int h = 0; h < hidden; h++) {
                double sum = 0.0d;
                for (int p = 0; p < ple; p++) sum += (double) projectionWeights[h][p] * gated[p];
                update[h] = (float) sum;
            }
            EmbeddingGemma2PerLayerEmbeddingProjector.rmsNormalize(update, normScale, epsilon);
            for (int h = 0; h < hidden; h++) output[token][h] = hiddenStates[token][h] + update[h];
        }
        return output;
    }

    private static float geluTanh(float value) {
        double x = value;
        double inner = Math.sqrt(2.0d / Math.PI) * (x + 0.044715d * x * x * x);
        return (float) (0.5d * x * (1.0d + Math.tanh(inner)));
    }

    private static int requireMatrix(float[][] matrix, String name) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) {
            throw new IllegalArgumentException(name + " must be a non-empty matrix");
        }
        int columns = matrix[0].length;
        for (float[] row : matrix) {
            if (row == null || row.length != columns) throw new IllegalArgumentException(name + " must be rectangular");
        }
        return columns;
    }

    private static void requireShape(float[][] matrix, int rows, int columns, String name) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
            }
        }
    }
}
