package tech.kayys.aljabr.backend.metal;

import static org.junit.jupiter.api.Assertions.assertSame;

import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

class MetalTensorStorageWrappingTest {

    @Test
    void wrappingCpuFallbackResultPreservesStorageIdentity() throws Exception {
        String previous = System.getProperty("aljabr.kernel.force.cpu");
        System.setProperty("aljabr.kernel.force.cpu", "true");
        try {
            MetalComputeBackend metal = new MetalComputeBackend();
            AtomicInteger releases = new AtomicInteger();
            TensorStorage storage = new TensorStorage() {
                @Override
                public long sizeBytes() {
                    return Float.BYTES;
                }

                @Override
                public void retain() {
                }

                @Override
                public void release() {
                    releases.incrementAndGet();
                }
            };
            DefaultTensor source = new DefaultTensor(
                    new Shape(1), DType.F32, DeviceType.CPU, storage, new CpuBackend());

            Method wrap = MetalComputeBackend.class.getDeclaredMethod("wrap", Tensor.class);
            wrap.setAccessible(true);
            Tensor wrapped = (Tensor) wrap.invoke(metal, source);

            assertSame(storage, ((DefaultTensor) wrapped).storage());
            assertSame(metal, wrapped.backend());
            wrapped.release();
            org.junit.jupiter.api.Assertions.assertEquals(1, releases.get());
        } finally {
            if (previous == null) {
                System.clearProperty("aljabr.kernel.force.cpu");
            } else {
                System.setProperty("aljabr.kernel.force.cpu", previous);
            }
        }
    }
}
