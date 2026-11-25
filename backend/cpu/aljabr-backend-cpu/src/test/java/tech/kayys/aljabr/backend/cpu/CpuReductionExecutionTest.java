package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CpuReductionExecutionTest {

    @Test
    void scalarReductionsPreserveExistingSemantics() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 2, 2);
                Tensor sum = backend.sum(input);
                Tensor mean = backend.mean(input);
                Tensor max = backend.max(input)) {
            assertArrayEquals(new long[] {1}, sum.shape().dims());
            assertArrayEquals(new float[] {10f}, sum.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] {2.5f}, mean.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] {4f}, max.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void axisReductionsPreserveKeepDimensionAndNegativeAxisBehavior() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 2, 2);
                Tensor rowSums = backend.sum(input, 1, true);
                Tensor columnMeans = backend.mean(input, -2, false)) {
            assertArrayEquals(new long[] {2, 1}, rowSums.shape().dims());
            assertArrayEquals(new float[] {3f, 7f}, rowSums.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new long[] {2}, columnMeans.shape().dims());
            assertArrayEquals(new float[] {2f, 3f}, columnMeans.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void softmaxAndLogSoftmaxRemainStableAcrossRows() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1000f, 1001f, -1000f, -999f}, backend, 2, 2);
                Tensor probabilities = backend.softmax(input, 1);
                Tensor logProbabilities = backend.logSoftmax(input, 1)) {
            assertArrayEquals(new float[] {
                    0.26894143f, 0.7310586f,
                    0.26894143f, 0.7310586f
            }, probabilities.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] {
                    -1.3132617f, -0.3132617f,
                    -1.3132617f, -0.3132617f
            }, logProbabilities.toFloatArray(), 1.0e-6f);
            assertEquals(1f, probabilities.toFloatArray()[0] + probabilities.toFloatArray()[1], 1.0e-6f);
        }
    }

    @Test
    void reductionsAndNormalizationReadTransposedInputsLogically() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1f, 2f, 3f, 4f}, backend, 2, 2);
                Tensor transposed = input.transpose(0, 1);
                Tensor rowSums = backend.sum(transposed, 1, false);
                Tensor probabilities = backend.softmax(transposed, 1)) {
            assertArrayEquals(new float[] {4f, 6f}, rowSums.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] {
                    0.11920292f, 0.8807971f,
                    0.11920292f, 0.8807971f
            }, probabilities.toFloatArray(), 1.0e-6f);
        }
    }
}
