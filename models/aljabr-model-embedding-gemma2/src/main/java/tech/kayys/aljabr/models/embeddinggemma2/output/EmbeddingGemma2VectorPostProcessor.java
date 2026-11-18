package tech.kayys.aljabr.models.embeddinggemma2.output;

import java.util.Arrays;
import java.util.Set;

/**
 * Applies EmbeddingGemma 2's supported Matryoshka truncation and final L2
 * normalization to one already-projected 768-dimensional embedding.
 *
 * <p>This class does not perform token pooling or the learned projection.
 * Those operations belong to the model forward path. Truncation happens
 * before normalization, as required to preserve unit norm at the selected
 * output dimension.</p>
 */
public final class EmbeddingGemma2VectorPostProcessor {
    public static final int NATIVE_DIMENSION = 768;
    private static final Set<Integer> SUPPORTED_DIMENSIONS = Set.of(768, 512, 256, 128);

    private EmbeddingGemma2VectorPostProcessor() {
    }

    /**
     * Truncates a native projected vector and L2-normalizes the retained
     * dimensions using float32 output and double-precision accumulation.
     *
     * @param projectedVector learned projection output, exactly 768 values
     * @param outputDimension requested Matryoshka dimension
     * @return a new normalized vector; the input is never modified
     */
    public static float[] process(float[] projectedVector, int outputDimension) {
        if (projectedVector == null) {
            throw new IllegalArgumentException("projectedVector must not be null");
        }
        if (projectedVector.length != NATIVE_DIMENSION) {
            throw new IllegalArgumentException(
                    "projectedVector must contain " + NATIVE_DIMENSION
                            + " values, got " + projectedVector.length);
        }
        if (!SUPPORTED_DIMENSIONS.contains(outputDimension)) {
            throw new IllegalArgumentException(
                    "outputDimension must be one of " + SUPPORTED_DIMENSIONS
                            + ", got " + outputDimension);
        }

        float[] result = Arrays.copyOf(projectedVector, outputDimension);
        double squaredNorm = 0.0d;
        for (int i = 0; i < result.length; i++) {
            float value = result[i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "projectedVector contains a non-finite value at index " + i);
            }
            squaredNorm += (double) value * value;
        }

        if (!Double.isFinite(squaredNorm) || squaredNorm <= 0.0d) {
            throw new IllegalArgumentException(
                    "projectedVector retained dimensions must have a finite, non-zero norm");
        }

        double inverseNorm = 1.0d / Math.sqrt(squaredNorm);
        for (int i = 0; i < result.length; i++) {
            result[i] = (float) (result[i] * inverseNorm);
        }
        return result;
    }

    public static boolean isSupportedDimension(int dimension) {
        return SUPPORTED_DIMENSIONS.contains(dimension);
    }
}
