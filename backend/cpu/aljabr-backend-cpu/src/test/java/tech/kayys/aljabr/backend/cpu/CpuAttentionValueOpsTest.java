package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuAttentionValueOpsTest {
    @Test
    void vectorizedAccumulationMatchesScalarAcrossVectorTails() {
        for (int dimension : new int[] {1, 7, 16, 31, 64, 80, 96, 128}) {
            float[] values = new float[37 * dimension];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.sin((i + 1) * 0.017);
            }
            float[] scalar = new float[dimension];
            float[] vectorized = new float[dimension];
            for (int key = 0; key < 37; key++) {
                float previousScale = 0.997f + (key % 3) * 0.001f;
                float currentScale = 0.001f + (key % 5) * 0.0001f;
                CpuAttentionValueOps.scalar(scalar, values, key * dimension,
                        previousScale, currentScale, dimension);
                CpuAttentionValueOps.vectorized(vectorized, values, key * dimension,
                        previousScale, currentScale, dimension);
            }
            assertArrayEquals(scalar, vectorized, 2e-6f,
                    "dimension " + dimension);
        }
    }

    @Test
    void f32CandidateStaysWithinBoundOfDoubleAccumulatorReference() {
        int dimension = 67;
        int keyCount = 1024;
        float[] values = new float[keyCount * dimension];
        float[] candidate = new float[dimension];
        double[] reference = new double[dimension];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.sin((i + 1) * 0.017);
        }

        for (int key = 0; key < keyCount; key++) {
            float previousScale = 0.997f + (key % 3) * 0.001f;
            float currentScale = 0.001f + (key % 5) * 0.0001f;
            CpuAttentionValueOps.vectorized(candidate, values, key * dimension,
                    previousScale, currentScale, dimension);
            for (int d = 0; d < dimension; d++) {
                reference[d] = reference[d] * previousScale
                        + (double) currentScale * values[key * dimension + d];
            }
        }

        double maxAbsoluteError = 0d;
        for (int d = 0; d < dimension; d++) {
            maxAbsoluteError = Math.max(maxAbsoluteError,
                    Math.abs(candidate[d] - reference[d]));
        }
        assertTrue(maxAbsoluteError < 2e-4,
                "max absolute error was " + maxAbsoluteError);
    }
}
