package tech.kayys.aljabr.models.embeddinggemma2.output;

import java.util.Set;
import tech.kayys.aljabr.models.embedding.common.L2Normalizer;

/**
 * EmbeddingGemma 2 output contract: validate the supported Matryoshka
 * dimensions, truncate, then delegate the reusable L2 normalization operation.
 */
public final class EmbeddingGemma2VectorPostProcessor {
    public static final int NATIVE_DIMENSION = 768;
    private static final Set<Integer> SUPPORTED_DIMENSIONS = Set.of(768, 512, 256, 128);

    private EmbeddingGemma2VectorPostProcessor() {
    }

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
        return L2Normalizer.normalizePrefix(projectedVector, outputDimension);
    }

    public static boolean isSupportedDimension(int dimension) {
        return SUPPORTED_DIMENSIONS.contains(dimension);
    }
}
