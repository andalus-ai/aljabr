package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.testing.tensor.TensorLifecycleAssertions;

/**
 * Applies the shared tensor lifecycle contract to tensors backed by the
 * production CPU backend and its off-heap storage implementation.
 */
class CpuTensorLifecycleConformanceTest {

    private static Tensor newTensor(CpuBackend backend) {
        return new DefaultTensor(
                new Shape(1),
                DType.F32,
                DeviceType.CPU,
                new CpuBuffer(Float.BYTES),
                backend);
    }

    @Test
    void cpuTensorReleaseIsIdempotent() {
        CpuBackend backend = new CpuBackend();
        try {
            TensorLifecycleAssertions.assertReleaseIsIdempotent(() -> newTensor(backend));
        } finally {
            backend.close();
        }
    }

    @Test
    void cpuTensorReleasesTransferredGradientExactlyOnce() {
        CpuBackend backend = new CpuBackend();
        try {
            TensorLifecycleAssertions.assertOwnedGradientReleasedExactlyOnce(
                    () -> newTensor(backend),
                    () -> newTensor(backend));
        } finally {
            backend.close();
        }
    }

    @Test
    void backendCloseDoesNotReplaceTensorStorageOwnership() {
        CpuBackend backend = new CpuBackend();
        Tensor tensor = newTensor(backend);

        backend.close();

        // Closing the backend must not implicitly invalidate a tensor whose
        // storage is independently owned by that tensor.
        assertArrayEquals(new float[] {0.0f}, tensor.toFloatArray());
        tensor.release();
    }
}
