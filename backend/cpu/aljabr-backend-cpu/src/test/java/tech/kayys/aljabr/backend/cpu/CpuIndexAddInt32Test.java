package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class CpuIndexAddInt32Test {

    @Test
    void accumulatesRepeatedIndicesAcrossOuterSlices() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new int[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor indices = TensorFactory.of(new int[] {1, 1, 0}, backend, 3);
                Tensor source = TensorFactory.of(new int[] {10, 20, 30, 40, 50, 60}, backend, 2, 3);
                Tensor result = destination.indexAdd(1, indices, source)) {
            assertArrayEquals(new int[] {31, 32, 3, 64, 95, 6}, result.toIntArray());
        }
    }

    @Test
    void readsNonContiguousDestinationAndSourceInLogicalOrder() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destinationBase = TensorFactory.of(new int[] {1, 2, 3, 4, 5, 6}, backend, 2, 3);
                Tensor destination = destinationBase.transpose(0, 1);
                Tensor indices = TensorFactory.of(new int[] {2, 0}, backend, 2);
                Tensor sourceBase = TensorFactory.of(new int[] {10, 30, 20, 40}, backend, 2, 2);
                Tensor source = sourceBase.transpose(0, 1);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertArrayEquals(new int[] {31, 44, 2, 5, 13, 26}, result.toIntArray());
        }
    }

    @Test
    void preservesTwoComplementIntegerAdditionSemantics() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = TensorFactory.of(new int[] {Integer.MAX_VALUE}, backend, 1);
                Tensor indices = TensorFactory.of(new int[] {0}, backend, 1);
                Tensor source = TensorFactory.of(new int[] {1}, backend, 1);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertArrayEquals(new int[] {Integer.MIN_VALUE}, result.toIntArray());
        }
    }
}
