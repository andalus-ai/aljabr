package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;

class CpuIndexSelectTest {

    @Test
    void selectsAxisEntriesPreservingOrderAndDuplicates() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor indices = TensorFactory.of(new int[] {2, 0, 2}, backend, 3);
                Tensor selected = input.indexSelect(-1, indices)) {
            assertEquals(DType.F32, selected.dtype());
            assertArrayEquals(new long[] {2, 3}, selected.shape().dims());
            assertArrayEquals(new float[] {3, 1, 3, 6, 4, 6}, selected.toFloatArray());
        }
    }

    @Test
    void selectsFromTransposedInputInLogicalOrder() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor transposed = input.transpose(0, 1);
                Tensor indices = TensorFactory.of(new int[] {2, 0}, backend, 2);
                Tensor selected = transposed.indexSelect(0, indices)) {
            assertArrayEquals(new long[] {2, 2}, selected.shape().dims());
            assertArrayEquals(new float[] {3, 6, 1, 4}, selected.toFloatArray());
        }
    }

    @Test
    void preservesI32ValuesAndNonContiguousLogicalOrder() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new int[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor transposed = input.transpose(0, 1);
                Tensor indices = TensorFactory.of(new int[] {2, 0}, backend, 2);
                Tensor selected = transposed.indexSelect(0, indices)) {
            assertEquals(DType.I32, selected.dtype());
            assertArrayEquals(new long[] {2, 2}, selected.shape().dims());
            assertArrayEquals(new int[] {3, 6, 1, 4}, selected.toIntArray());
        }
    }

    @Test
    void emptyIndicesProduceAnEmptySelectedAxis() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor indices = TensorFactory.of(new int[0], backend, 0);
                Tensor selected = input.indexSelect(1, indices)) {
            assertArrayEquals(new long[] {2, 0}, selected.shape().dims());
            assertArrayEquals(new float[0], selected.toFloatArray());
        }
    }

    @Test
    void rejectsOutOfBoundsIndicesAndNonI32IndexTensors() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor tooLarge = TensorFactory.of(new int[] {2}, backend, 1);
                Tensor negative = TensorFactory.of(new int[] {-1}, backend, 1);
                Tensor floatIndices = TensorFactory.of(new float[] {1}, backend, 1)) {
            assertThrows(IndexOutOfBoundsException.class, () -> input.indexSelect(1, tooLarge));
            assertThrows(IndexOutOfBoundsException.class, () -> input.indexSelect(1, negative));
            assertThrows(IllegalArgumentException.class, () -> input.indexSelect(1, floatIndices));
        }
    }

    @Test
    void rejectsNonVectorIndicesAndInvalidAxes() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor matrixIndices = TensorFactory.of(new int[] {0}, backend, 1, 1);
                Tensor vectorIndices = TensorFactory.of(new int[] {0}, backend, 1)) {
            assertThrows(IllegalArgumentException.class, () -> input.indexSelect(0, matrixIndices));
            assertThrows(IllegalArgumentException.class, () -> input.indexSelect(2, vectorIndices));
        }
    }

    @Test
    void differentiableSelectionAccumulatesRepeatedIndexGradients() {
        try (CpuBackend backend = new CpuBackend();
                Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor indices = TensorFactory.of(new int[] {1, 1}, backend, 2)) {
            input.setRequiresGrad(true);
            try (Tensor selected = input.indexSelect(1, indices);
                    Tensor seed = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2)) {
                selected.backward(seed);
                assertArrayEquals(new float[] {0, 3, 0, 7}, input.grad().toFloatArray(), 1.0e-6f);
            }
        }
    }
}
