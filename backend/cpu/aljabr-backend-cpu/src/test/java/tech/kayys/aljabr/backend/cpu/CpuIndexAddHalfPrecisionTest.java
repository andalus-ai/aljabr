package tech.kayys.aljabr.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class CpuIndexAddHalfPrecisionTest {

    @Test
    void accumulatesRepeatedIndicesForF16() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = tensor16(backend, DType.F16, new short[] {
                        CpuFloatingPointConversions.floatToHalf(1),
                        CpuFloatingPointConversions.floatToHalf(2),
                        CpuFloatingPointConversions.floatToHalf(3)
                }, 3);
                Tensor indices = TensorFactory.of(new int[] {1, 1, 0}, backend, 3);
                Tensor source = tensor16(backend, DType.F16, new short[] {
                        CpuFloatingPointConversions.floatToHalf(10),
                        CpuFloatingPointConversions.floatToHalf(20),
                        CpuFloatingPointConversions.floatToHalf(30)
                }, 3);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertValues(result, DType.F16, 31f, 32f, 3f);
        }
    }

    @Test
    void accumulatesRepeatedIndicesForBF16() {
        try (CpuBackend backend = new CpuBackend();
                Tensor destination = tensor16(backend, DType.BF16, new short[] {
                        CpuFloatingPointConversions.floatToBfloat16(1),
                        CpuFloatingPointConversions.floatToBfloat16(2),
                        CpuFloatingPointConversions.floatToBfloat16(3)
                }, 3);
                Tensor indices = TensorFactory.of(new int[] {1, 1, 0}, backend, 3);
                Tensor source = tensor16(backend, DType.BF16, new short[] {
                        CpuFloatingPointConversions.floatToBfloat16(10),
                        CpuFloatingPointConversions.floatToBfloat16(20),
                        CpuFloatingPointConversions.floatToBfloat16(30)
                }, 3);
                Tensor result = destination.indexAdd(0, indices, source)) {
            assertValues(result, DType.BF16, 31f, 32f, 3f);
        }
    }

    private static Tensor tensor16(CpuBackend backend, DType dtype, short[] values, long... shape) {
        long bytes = dtype.memoryFootprintBytes(values.length);
        TensorStorage storage = backend.allocateStorage(bytes);
        try {
            backend.copyFromHost(storage, MemorySegment.ofArray(values), bytes);
            return new DefaultTensor(new Shape(shape), dtype, DeviceType.CPU, storage, backend);
        } catch (RuntimeException | Error failure) {
            storage.release();
            throw failure;
        }
    }

    private static void assertValues(Tensor tensor, DType dtype, float... expected) {
        HostAccessibleStorage storage = (HostAccessibleStorage) tensor.storage();
        MemorySegment segment = storage.segment();
        float[] actual = new float[expected.length];
        for (int i = 0; i < actual.length; i++) {
            short bits = segment.get(ValueLayout.JAVA_SHORT, (long) i * Short.BYTES);
            actual[i] = dtype == DType.F16
                    ? CpuFloatingPointConversions.halfToFloat(bits)
                    : CpuFloatingPointConversions.bfloat16ToFloat(bits);
        }
        assertArrayEquals(expected, actual, 0.001f);
    }
}
