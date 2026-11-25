package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class CpuNormalizationExecutionTest {

    @Test
    void layerNormNormalizesTrailingDimensions() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 3, 2, 4}, backend, 2, 2);
                Tensor output = backend.layerNorm(input, new long[] {2}, null, null, 0f)) {
            assertArrayEquals(new float[] {-1, 1, -1, 1}, output.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void rmsNormScalesRowsByRootMeanSquare() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor weight = TensorFactory.of(new float[] {1, 1}, backend, 2);
                Tensor output = backend.rmsNorm(input, weight, 0f)) {
            assertArrayEquals(new float[] {
                    0.6324555f, 1.264911f,
                    0.8485281f, 1.1313709f
            }, output.toFloatArray(), 1.0e-5f);
        }
    }

    @Test
    void batchNormUsesBatchAndSpatialStatisticsPerChannel() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor output = backend.batchNorm(input, null, null, null, null, false, 0.1f, 0f)) {
            assertArrayEquals(new float[] {-1, -1, 1, 1}, output.toFloatArray(), 1.0e-6f);
        }
    }

    @Test
    void batchNormUpdatesRunningStatisticsDuringTraining() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor runningMean = TensorFactory.of(new float[] {0, 0}, backend, 2);
                Tensor runningVar = TensorFactory.of(new float[] {1, 1}, backend, 2);
                Tensor output = backend.batchNorm(input, null, null, runningMean, runningVar, true, 0.5f, 0f)) {
            assertArrayEquals(new float[] {-1, -1, 1, 1}, output.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] {1, 1.5f}, runningMean.toFloatArray(), 1.0e-6f);
            assertArrayEquals(new float[] {1.5f, 1.5f}, runningVar.toFloatArray(), 1.0e-6f);
        }
    }
}
