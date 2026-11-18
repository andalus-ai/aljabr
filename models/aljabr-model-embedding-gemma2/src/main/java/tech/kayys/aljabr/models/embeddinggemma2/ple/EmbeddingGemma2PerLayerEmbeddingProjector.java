package tech.kayys.aljabr.models.embeddinggemma2.ple;

/**
 * Float32 reference for EmbeddingGemma 2's projection-only Per-Layer Embeddings.
 *
 * <p>The checkpoint projection is bias-free. Its output is scaled by
 * {@code hiddenSize^-0.5}, reshaped to [sequence][layer][PLE dimension], then
 * RMS-normalized independently for every token/layer slice. There is no
 * token-identity lookup table and no blending with a second PLE source.
 */
public final class EmbeddingGemma2PerLayerEmbeddingProjector {

    private EmbeddingGemma2PerLayerEmbeddingProjector() {
    }

    /**
     * @param inputEmbeddings [sequence][hiddenSize], already scaled token embeddings
     * @param projectionWeights checkpoint matrix [numLayers * pleSize][hiddenSize]
     * @param normScale checkpoint RMSNorm scale [pleSize]
     * @param numLayers number of transformer layers
     * @param pleSize hidden_size_per_layer_input
     * @param epsilon RMSNorm epsilon
     * @return [sequence][numLayers][pleSize]
     */
    public static float[][][] project(float[][] inputEmbeddings, float[][] projectionWeights,
            float[] normScale, int numLayers, int pleSize, float epsilon) {
        validate(inputEmbeddings, projectionWeights, normScale, numLayers, pleSize, epsilon);
        int sequenceLength = inputEmbeddings.length;
        int hiddenSize = inputEmbeddings[0].length;
        float scale = (float) (1.0d / Math.sqrt(hiddenSize));
        float[][][] result = new float[sequenceLength][numLayers][pleSize];

        for (int token = 0; token < sequenceLength; token++) {
            for (int layer = 0; layer < numLayers; layer++) {
                float[] projected = result[token][layer];
                int rowOffset = layer * pleSize;
                for (int dimension = 0; dimension < pleSize; dimension++) {
                    double sum = 0.0d;
                    float[] weightRow = projectionWeights[rowOffset + dimension];
                    for (int input = 0; input < hiddenSize; input++) {
                        sum += (double) weightRow[input] * inputEmbeddings[token][input];
                    }
                    projected[dimension] = (float) (sum * scale);
                }
                rmsNormalize(projected, normScale, epsilon);
            }
        }
        return result;
    }

    private static void validate(float[][] input, float[][] weights, float[] normScale,
            int layers, int pleSize, float epsilon) {
        if (input == null || input.length == 0 || input[0] == null || input[0].length == 0) {
            throw new IllegalArgumentException("inputEmbeddings must be a non-empty [sequence][hidden] matrix");
        }
        if (layers <= 0 || pleSize <= 0) {
            throw new IllegalArgumentException("numLayers and pleSize must be positive");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        int hidden = input[0].length;
        for (float[] token : input) {
            if (token == null || token.length != hidden) {
                throw new IllegalArgumentException("inputEmbeddings must be rectangular");
            }
            requireFinite(token, "inputEmbeddings");
        }
        if (weights == null || weights.length != Math.multiplyExact(layers, pleSize)) {
            throw new IllegalArgumentException("projectionWeights must have shape [numLayers * pleSize][hiddenSize]");
        }
        for (float[] row : weights) {
            if (row == null || row.length != hidden) {
                throw new IllegalArgumentException("projectionWeights must have shape [numLayers * pleSize][hiddenSize]");
            }
            requireFinite(row, "projectionWeights");
        }
        if (normScale == null || normScale.length != pleSize) {
            throw new IllegalArgumentException("normScale must have dimension pleSize");
        }
        requireFinite(normScale, "normScale");
    }

    static void rmsNormalize(float[] vector, float[] scale, float epsilon) {
        double meanSquare = 0.0d;
        for (float value : vector) meanSquare += (double) value * value;
        meanSquare = meanSquare / vector.length + epsilon;
        double inverseRms = 1.0d / Math.sqrt(meanSquare);
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (vector[i] * inverseRms * scale[i]);
        }
    }

    static void requireFinite(float[] values, String name) {
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException(name + " must contain finite values");
        }
    }
}
