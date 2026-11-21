package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

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

    private DefaultTensor tensor(Shape shape, float... values) {
        CpuBuffer buffer = new CpuBuffer((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            buffer.segment().set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, values[i]);
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, buffer, backend);
    }

    @Test
    void splitRejectsInvalidAxisAndPartCounts() {
        DefaultTensor input = tensor(new Shape(2, 4), 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, -3, 2));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 2, 2));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, 0));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, -1));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, 5));
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 1, 3));
        } finally {
            input.release();
        }
    }

    @Test
    void splitProducesEqualNonOverlappingViews() {
        DefaultTensor input = tensor(new Shape(6), 1f, 2f, 3f, 4f, 5f, 6f);
        List<Tensor> chunks = null;
        try {
            chunks = backend.split(input, 0, 3);
            assertEquals(3, chunks.size());
            assertArrayEquals(new float[] { 1f, 2f }, chunks.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[] { 3f, 4f }, chunks.get(1).toFloatArray(), 0f);
            assertArrayEquals(new float[] { 5f, 6f }, chunks.get(2).toFloatArray(), 0f);
        } finally {
            if (chunks != null) {
                for (Tensor chunk : chunks) chunk.close();
            }
            input.release();
        }
    }

    @Test
    void splitNormalizesNegativeAxisAndPreservesChunkOrder() {
        DefaultTensor input = tensor(new Shape(2, 6),
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f);
        List<Tensor> chunks = null;
        try {
            chunks = backend.split(input, -1, 3);
            assertEquals(3, chunks.size());
            assertArrayEquals(new long[] { 2, 2 }, chunks.get(0).shape().dims());
            assertArrayEquals(new float[] { 1f, 2f, 7f, 8f }, chunks.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[] { 3f, 4f, 9f, 10f }, chunks.get(1).toFloatArray(), 0f);
            assertArrayEquals(new float[] { 5f, 6f, 11f, 12f }, chunks.get(2).toFloatArray(), 0f);
        } finally {
            if (chunks != null) {
                for (Tensor chunk : chunks) chunk.close();
            }
            input.release();
        }
    }

    @Test
    void tensorSplitUsesSharedNegativeAxisValidation() {
        DefaultTensor input = tensor(new Shape(2, 6),
                1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f);
        List<Tensor> chunks = null;
        try {
            chunks = input.split(-1, 3);
            assertEquals(3, chunks.size());
            assertArrayEquals(new float[] { 1f, 2f, 7f, 8f }, chunks.get(0).toFloatArray(), 0f);
            assertArrayEquals(new float[] { 3f, 4f, 9f, 10f }, chunks.get(1).toFloatArray(), 0f);
            assertArrayEquals(new float[] { 5f, 6f, 11f, 12f }, chunks.get(2).toFloatArray(), 0f);
        } finally {
            if (chunks != null) chunks.forEach(Tensor::close);
            input.release();
        }
    }

    @Test
    void splitRejectsEmptyAxis() {
        CpuBuffer buffer = new CpuBuffer(0);
        DefaultTensor input = new DefaultTensor(
                new Shape(0), DType.F32, DeviceType.CPU, buffer, backend);
        try {
            assertThrows(IllegalArgumentException.class, () -> backend.split(input, 0, 1));
        } finally {
            input.release();
        }
    }
}
