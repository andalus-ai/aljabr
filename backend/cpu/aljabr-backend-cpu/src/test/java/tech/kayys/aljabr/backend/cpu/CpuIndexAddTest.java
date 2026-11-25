package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class CpuIndexAddTest {

    @Test
    void accumulatesRepeatedIndicesAcrossOuterSlices() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new float[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor indices = TensorFactory.of(new int[] {1, 1, 0}, backend, 3);
                Tensor source = TensorFactory.of(new float[] {10, 20, 30, 40, 50, 60}, backend, 2, 3);
                Tensor result = destination.indexAdd(1, indices, source)) {
            assertArrayEquals(new float[] {31, 32, 3, 64, 95, 6}, result.toFloatArray());
        }
    }

    @Test
    void readsNonContiguousDestinationAndSourceInLogicalOrder() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destinationBase = TensorFactory.of(new float[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor destination = destinationBase.transpose(0, 1);
                Tensor indices = TensorFactory.of(new int[] {2, 0}, backend, 2);
                Tensor sourceBase = TensorFactory.of(new float[] {10, 30, 20, 40}, backend, 2, 2);
                Tensor source = sourceBase.transpose(0, 1);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertArrayEquals(new float[] {31, 44, 2, 5, 13, 26}, result.toFloatArray());
        }
    }

    @Test
    void emptyIndicesReturnAnIndependentCopyOfDestination() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor indices = TensorFactory.of(new int[0], backend, 0);
                Tensor source = TensorFactory.of(new float[0], backend, 2, 0);
                Tensor result = destination.indexAdd(1, indices, source)) {
            assertArrayEquals(new float[] {1, 2, 3, 4}, result.toFloatArray());
        }
    }

    @Test
    void rejectsNegativeAndOutOfRangeIndicesAndMismatchedSourceShape() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor tooLarge = TensorFactory.of(new int[] {2}, backend, 1);
                Tensor negative = TensorFactory.of(new int[] {-1}, backend, 1);
                Tensor validIndices = TensorFactory.of(new int[] {0}, backend, 1);
                Tensor validSource = TensorFactory.of(new float[] {10, 20}, backend, 2, 1);
                Tensor wrongSource = TensorFactory.of(new float[] {10, 20, 30, 40}, backend, 2, 2)) {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> destination.indexAdd(1, tooLarge, validSource));
            assertThrows(IndexOutOfBoundsException.class,
                    () -> destination.indexAdd(1, negative, validSource));
            assertThrows(IllegalArgumentException.class,
                    () -> destination.indexAdd(1, validIndices, wrongSource));
        }
    }

    @Test
    void rejectsSourceWithDifferentDtypeBeforeBackendExecution() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor indices = TensorFactory.of(new int[] {0}, backend, 1);
                Tensor integerSource = TensorFactory.of(new int[] {10, 20}, backend, 2, 1)) {
            assertThrows(IllegalArgumentException.class,
                    () -> destination.indexAdd(1, indices, integerSource));
        }
    }

    @Test
    void rejectsNonVectorIndices() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 2, 2);
                Tensor matrixIndices = TensorFactory.of(new int[] {0}, backend, 1, 1);
                Tensor source = TensorFactory.of(new float[] {10, 20}, backend, 2, 1)) {
            assertThrows(IllegalArgumentException.class,
                    () -> destination.indexAdd(1, matrixIndices, source));
        }
    }
}
