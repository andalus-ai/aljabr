package tech.kayys.aljabr.models.embedding.common;

import java.util.Arrays;

/** Stateless float32 L2 normalization utilities for embedding vectors. */
public final class L2Normalizer {
    private L2Normalizer() {
    }

    /** Returns a normalized copy of the complete vector. */
    public static float[] normalize(float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("vector must not be null");
        }
        return normalizePrefix(vector, vector.length);
    }

    /**
     * Truncates to {@code outputDimension}, then L2-normalizes the retained
     * prefix. This ordering supports Matryoshka-style embedding dimensions.
     */
    public static float[] normalizePrefix(float[] vector, int outputDimension) {
        if (vector == null) {
            throw new IllegalArgumentException("vector must not be null");
        }
        if (outputDimension <= 0 || outputDimension > vector.length) {
            throw new IllegalArgumentException(
                    "outputDimension must be between 1 and vector length " + vector.length);
        }

        float[] result = Arrays.copyOf(vector, outputDimension);
        double squaredNorm = 0.0d;
        for (int index = 0; index < result.length; index++) {
            float value = result[index];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("vector contains a non-finite value at index " + index);
            }
            squaredNorm += (double) value * value;
        }
        if (!Double.isFinite(squaredNorm) || squaredNorm <= 0.0d) {
            throw new IllegalArgumentException("vector must have a finite, non-zero L2 norm");
        }

        double inverseNorm = 1.0d / Math.sqrt(squaredNorm);
        for (int index = 0; index < result.length; index++) {
            result[index] = (float) (result[index] * inverseNorm);
        }
        return result;
    }
}
