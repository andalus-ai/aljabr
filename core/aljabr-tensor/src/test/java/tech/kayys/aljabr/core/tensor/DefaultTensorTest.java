package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;

class DefaultTensorTest {
    @Test
    void itemReturnsActualScalarValue() {
        CpuBuffer buffer = new CpuBuffer(Float.BYTES);
        buffer.segment().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 3.5f);
        DefaultTensor tensor = new DefaultTensor(new Shape(1), DType.F32, DeviceType.CPU, buffer, null);
        try {
            assertEquals(3.5f, tensor.item());
        } finally {
            tensor.release();
        }
    }

    @Test
    void itemRejectsNonScalarTensors() {
        CpuBuffer buffer = new CpuBuffer(2L * Float.BYTES);
        DefaultTensor tensor = new DefaultTensor(new Shape(2), DType.F32, DeviceType.CPU, buffer, null);
        try {
            assertThrows(IllegalStateException.class, tensor::item);
        } finally {
            tensor.release();
        }
    }

    @Test
    void floatArrayExportRejectsQuantizedDataInsteadOfReinterpretingBytes() {
        CpuBuffer buffer = new CpuBuffer(18);
        DefaultTensor tensor = new DefaultTensor(new Shape(32), DType.Q4_0, DeviceType.CPU, buffer, null);
        try {
            assertThrows(UnsupportedOperationException.class, tensor::toFloatArray);
        } finally {
            tensor.release();
        }
    }
}
