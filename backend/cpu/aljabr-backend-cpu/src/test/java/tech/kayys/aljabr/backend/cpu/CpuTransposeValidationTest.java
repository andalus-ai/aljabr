package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Tensor;

class CpuTransposeValidationTest {

    private final CpuBackend backend = new CpuBackend();

    @Test
    void transposeSwapsRectangularMatrixDimensionsAndValues() {
        DefaultTensor input = new DefaultTensor(
                new tech.kayys.aljabr.core.tensor.Shape(2, 3), DType.F32, DeviceType.CPU,
                floats(1, 2, 3, 4, 5, 6), backend);
        try (Tensor transposed = backend.transpose(input)) {
            assertArrayEquals(new long[] {3, 2}, transposed.shape().dims());
            assertArrayEquals(new float[] {1, 4, 2, 5, 3, 6}, transposed.toFloatArray(), 0f);
        } finally {
            input.close();
        }
    }

    @Test
    void transposeHonorsNonZeroStorageOffsetAndParentStrides() {
        DefaultTensor input = new DefaultTensor(
                new tech.kayys.aljabr.core.tensor.Shape(2, 3), DType.F32, DeviceType.CPU,
                floats(1, 2, 3, 4, 5, 6), backend);
        Tensor view = backend.slice(input, new long[] {0, 1}, new long[] {2, 2});
        try (Tensor transposed = backend.transpose(view, 0, 1)) {
            assertArrayEquals(new long[] {2, 2}, transposed.shape().dims());
            assertArrayEquals(new float[] {2, 5, 3, 6}, transposed.toFloatArray(), 0f);
        } finally {
            view.close();
            input.close();
        }
    }

    @Test
    void transposeRejectsInvalidDimensionsAndDefaultRankBelowTwo() {
        DefaultTensor matrix = new DefaultTensor(
                new tech.kayys.aljabr.core.tensor.Shape(2, 2), DType.F32, DeviceType.CPU,
                floats(1, 2, 3, 4), backend);
        DefaultTensor vector = new DefaultTensor(
                new tech.kayys.aljabr.core.tensor.Shape(4), DType.F32, DeviceType.CPU,
                floats(1, 2, 3, 4), backend);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.transpose(matrix, -1, 1));
            assertThrows(IllegalArgumentException.class, () -> backend.transpose(matrix, 0, 2));
            assertThrows(IllegalArgumentException.class, () -> backend.transpose(vector));
        } finally {
            matrix.close();
            vector.close();
        }
    }

    private CpuBuffer floats(float... values) {
        CpuBuffer buffer = new CpuBuffer((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            buffer.segment().set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
        return buffer;
    }
}
