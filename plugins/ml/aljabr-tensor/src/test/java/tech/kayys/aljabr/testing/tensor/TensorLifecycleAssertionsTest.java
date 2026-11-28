package tech.kayys.aljabr.testing.tensor;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;

class TensorLifecycleAssertionsTest {

    private static DefaultTensor tensor() {
        return new DefaultTensor(
                new Shape(1),
                DType.F32,
                DeviceType.CPU,
                new CpuBuffer(Float.BYTES),
                new CpuBackend());
    }

    @Test
    void verifiesReleaseAndCloseAreIdempotent() {
        TensorLifecycleAssertions.assertReleaseIsIdempotent(TensorLifecycleAssertionsTest::tensor);
    }

    @Test
    void verifiesTransferredGradientIsReleasedExactlyOnce() {
        TensorLifecycleAssertions.assertOwnedGradientReleasedExactlyOnce(
                TensorLifecycleAssertionsTest::tensor,
                TensorLifecycleAssertionsTest::tensor);
    }
}
