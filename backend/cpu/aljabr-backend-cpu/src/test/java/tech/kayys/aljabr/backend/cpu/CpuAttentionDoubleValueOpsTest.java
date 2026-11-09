package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuAttentionDoubleValueOpsTest {
    @Test
    void vectorizedAccumulationMatchesScalarAcrossVectorTails() {
        for (int dimension : new int[] {1, 7, 16, 31, 64, 80, 96, 128}) {
            double[] values = new double[37 * dimension];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.sin((i + 1) * 0.017);
            }
            double[] scalar = new double[dimension];
            double[] vectorized = new double[dimension];
            for (int key = 0; key < 37; key++) {
                double previousScale = 0.997d + (key % 3) * 0.001d;
                double currentScale = 0.001d + (key % 5) * 0.0001d;
                CpuAttentionDoubleValueOps.scalar(scalar, values, key * dimension,
                        previousScale, currentScale, dimension);
                CpuAttentionDoubleValueOps.vectorized(vectorized, values, key * dimension,
                        previousScale, currentScale, dimension);
            }
            assertArrayEquals(scalar, vectorized, 1e-12,
                    "dimension " + dimension);
        }
    }
}
