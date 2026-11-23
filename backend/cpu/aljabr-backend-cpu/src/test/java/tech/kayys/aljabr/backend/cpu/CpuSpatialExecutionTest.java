package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class CpuSpatialExecutionTest {

    @Test
    void convolutionUsesNchwInputAndOihwWeights() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9
                }, backend, 1, 1, 3, 3);
                Tensor weight = TensorFactory.of(new float[] {1, 1, 1, 1}, backend, 1, 1, 2, 2);
                Tensor output = backend.conv2d(input, weight, null, 1, 0, 1, 1)) {
            assertArrayEquals(new long[] {1, 1, 2, 2}, output.shape().dims());
            assertArrayEquals(new float[] {12, 16, 24, 28}, output.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void nearestUpsamplingUsesFloorScaledSourceCoordinates() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 1, 1, 2, 2);
                Tensor output = backend.upsampleNearest2d(input, 4, 4)) {
            assertArrayEquals(new long[] {1, 1, 4, 4}, output.shape().dims());
            assertArrayEquals(new float[] {
                    1, 1, 2, 2,
                    1, 1, 2, 2,
                    3, 3, 4, 4,
                    3, 3, 4, 4
            }, output.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void maxPoolUsesWindowMaximum() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {
                        1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12,
                        13, 14, 15, 16
                }, backend, 1, 1, 4, 4);
                Tensor output = backend.maxPool2d(input, 2, 2, 0)) {
            assertArrayEquals(new long[] {1, 1, 2, 2}, output.shape().dims());
            assertArrayEquals(new float[] {6, 8, 14, 16}, output.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void adaptiveAveragePoolPartitionsUnevenSpatialDimensions() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9
                }, backend, 1, 1, 3, 3);
                Tensor output = backend.adaptiveAvgPool2d(input, 2, 2)) {
            assertArrayEquals(new long[] {1, 1, 2, 2}, output.shape().dims());
            assertArrayEquals(new float[] {3, 4, 6, 7}, output.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void spatialKernelsReadTransposedInputInLogicalOrder() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 1, 1, 2, 2);
                Tensor transposed = input.transpose(2, 3);
                Tensor output = backend.maxPool2d(transposed, 2, 1, 0)) {
            assertArrayEquals(new float[] {4}, output.toFloatArray(), 1.0e-6f);
        }
    }
}
