package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class CpuIndexAddInt8Test {

    @Test
    void accumulatesRepeatedIndicesForI8() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = Tensor.of(new byte[] {1, 2, 3, 4, 5, 6}, backend, DType.I8, 2, 3);
                Tensor indices = TensorFactory.of(new int[] {1, 1, 0}, backend, 3);
                Tensor source = Tensor.of(new byte[] {10, 20, 30, 40, 50, 60}, backend, DType.I8, 2, 3);
                Tensor result = destination.indexAdd(1, indices, source)) {
            assertArrayEquals(new byte[] {31, 32, 3, 64, 95, 6}, result.toByteArray());
            assertArrayEquals(new byte[] {1, 2, 3, 4, 5, 6}, destination.toByteArray());
        }
    }

    @Test
    void supportsInt8AliasAndSignedWraparound() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = Tensor.of(new byte[] {Byte.MAX_VALUE}, backend, DType.INT8, 1);
                Tensor indices = TensorFactory.of(new int[] {0}, backend, 1);
                Tensor source = Tensor.of(new byte[] {1}, backend, DType.INT8, 1);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertArrayEquals(new byte[] {Byte.MIN_VALUE}, result.toByteArray());
        }
    }

    @Test
    void readsNonContiguousDestinationAndSourceInLogicalOrder() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destinationBase = Tensor.of(new byte[] {1, 2, 3, 4, 5, 6}, backend, DType.I8, 2, 3);
                Tensor destination = destinationBase.transpose(0, 1);
                Tensor indices = TensorFactory.of(new int[] {2, 0}, backend, 2);
                Tensor sourceBase = Tensor.of(new byte[] {10, 30, 20, 40}, backend, DType.I8, 2, 2);
                Tensor source = sourceBase.transpose(0, 1);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertArrayEquals(new byte[] {31, 44, 2, 5, 13, 26}, result.toByteArray());
        }
    }

    @Test
    void rejectsByteTensorFactoryForOtherDtypes() {
        try (CpuBackend backend = new CpuBackend()) {
            assertThrows(IllegalArgumentException.class,
                    () -> Tensor.of(new byte[] {1}, backend, DType.I32, 1));
        }
    }
}
