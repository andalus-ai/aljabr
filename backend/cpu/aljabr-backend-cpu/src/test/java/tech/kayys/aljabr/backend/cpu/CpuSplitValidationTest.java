package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.ValueLayout;
import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

class CpuSplitValidationTest {

    private final CpuBackend backend = new CpuBackend();

    @Test
    void splitNormalizesNegativeAxisAndReturnsOrderedSlices() {
        DefaultTensor input = tensor(new Shape(2, 6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        List<Tensor> outputs = null;
        try {
            outputs = backend.split(input, -1, 3);
            assertEquals(3, outputs.size());
            assertArrayEquals(new long[] {2, 2}, outputs.get(0).shape().dims());
            assertArrayEquals(new float[] {1, 2, 7, 8}, outputs.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[] {3, 4, 9, 10}, outputs.get(1).toFloatArray(), 0f);
            assertArrayEquals(new float[] {5, 6, 11, 12}, outputs.get(2).toFloatArray(), 0f);
        } finally {
            if (outputs != null) outputs.forEach(Tensor::close);
            input.close();
        }
    }

    @Test
    void splitRejectsInvalidAxisAndNonDivisibleOrEmptyDimensions() {
        DefaultTensor input = tensor(new Shape(2, 3), 1, 2, 3, 4, 5, 6);
        DefaultTensor empty = tensor(new Shape(2, 0));
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, -3, 1));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 2, 1));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, 2));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, 4));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, 0));
            assertThrows(IllegalArgumentException.class, () -> backend.split(empty, 1, 1));
        } finally {
            input.close();
            empty.close();
        }
    }

    private static DefaultTensor tensor(Shape shape, float... values) {
        CpuBuffer buffer = new CpuBuffer((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            buffer.segment().set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, buffer, new CpuBackend());
    }
}
