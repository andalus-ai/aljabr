package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.ResourceLock;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class MetalRotaryEmbeddingFallbackTest {

    @Test
    @ResourceLock("system-properties")
    void configurableRoPEUsesCpuFallbackAndPreservesBackendIdentity() {
        String previous = System.getProperty("aljabr.kernel.force.cpu");
        System.setProperty("aljabr.kernel.force.cpu", "true");
        MetalComputeBackend backend = new MetalComputeBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f, 9f, 9f}, backend, 1, 1, 1, 4);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 2, 10_000f, 2, false, 1f);
            assertArrayEquals(new float[] {(float) Math.cos(2.0), (float) Math.sin(2.0), 9f, 9f},
                    output.toFloatArray(), 1e-5f);
            assertSame(backend, output.backend());
        } finally {
            if (output != null) output.release();
            input.release();
            if (previous == null) System.clearProperty("aljabr.kernel.force.cpu");
            else System.setProperty("aljabr.kernel.force.cpu", previous);
        }
    }

    @Test
    @ResourceLock("system-properties")
    void explicitPositionIdsUseCpuFallbackForBatchSpecificPositions() {
        String previous = System.getProperty("aljabr.kernel.force.cpu");
        System.setProperty("aljabr.kernel.force.cpu", "true");
        MetalComputeBackend backend = new MetalComputeBackend();
        Tensor input = TensorFactory.of(new float[] {1f, 0f, 1f, 0f}, backend, 2, 1, 1, 2);
        Tensor output = null;
        try {
            output = backend.applyRoPE(input, 10_000f, 2, false, 1f, new long[] {0L, 1L});
            assertArrayEquals(new float[] {1f, 0f, (float) Math.cos(1.0), (float) Math.sin(1.0)},
                    output.toFloatArray(), 1e-5f);
            assertSame(backend, output.backend());
        } finally {
            if (output != null) output.release();
            input.release();
            if (previous == null) System.clearProperty("aljabr.kernel.force.cpu");
            else System.setProperty("aljabr.kernel.force.cpu", previous);
        }
    }
}
