package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Proxy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.CpuBuffer;

class DefaultTensorTest {
    private static ComputeBackend unusedBackend() {
        return (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    throw new UnsupportedOperationException("Backend operations are not used by this test");
                });
    }

    @Test
    void itemReturnsActualScalarValue() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        buffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 3.5f);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            assertEquals(3.5f, tensor.item(), 0f);
        } finally {
            tensor.release();
        }
    }

    @Test
    void itemRejectsNonScalarTensors() {
        CpuBuffer buffer = new CpuBuffer(2L * Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(2), DType.F32, DeviceType.CPU, buffer, unusedBackend());
        try {
            assertThrows(IllegalStateException.class, tensor::item);
        } finally {
            tensor.release();
        }
    }

    @Test
    void floatArrayExportRejectsQuantizedDataInsteadOfReinterpretingBytes() {
        CpuBuffer buffer = new CpuBuffer(18);
        DefaultTensor tensor = new DefaultTensor(new Shape(32), DType.Q4_0, DeviceType.CPU, buffer, unusedBackend());
        try {
            assertThrows(UnsupportedOperationException.class, tensor::toFloatArray);
        } finally {
            tensor.release();
        }
    }
}
