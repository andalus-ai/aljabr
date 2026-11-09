package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.memory.CpuBuffer;

class TensorFactoryTest {
    @Test
    void rejectsShapesTooLargeForJavaArraysBeforeAllocation() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.zeros(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.ones(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.randn(Long.MAX_VALUE));
    }

    @Test
    void rejectsDataWhoseLengthDoesNotMatchShape() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.of(new float[] {1f}, 2));
    }

    @Test
    void rejectsQuantizedStorageWithIncompleteBlockShape() {
        var segment = Arena.ofAuto().allocate(16);
        assertThrows(IllegalArgumentException.class,
                () -> TensorFactory.ofQuantized(segment, DType.Q8_0, 31));
    }

    @Test
    void quantizedTensorOwnsACopyOfTheRequiredStorageBytes() {
        try (Arena sourceArena = Arena.ofConfined()) {
            MemorySegment source = sourceArena.allocate(36);
            for (long i = 0; i < source.byteSize(); i++) {
                source.set(java.lang.foreign.ValueLayout.JAVA_BYTE, i, (byte) (i + 1));
            }

            Tensor tensor = TensorFactory.ofQuantized(source, DType.Q8_0, 32);
            try {
                assertEquals(DType.Q8_0, tensor.dtype());
                assertEquals(34, tensor.storage().sizeBytes());
                assertInstanceOf(CpuBuffer.class, tensor.storage());
                MemorySegment copied = ((CpuBuffer) tensor.storage()).segment();
                for (long i = 0; i < 34; i++) {
                    assertEquals(source.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i),
                            copied.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i));
                }
            } finally {
                tensor.release();
            }
        }
    }

    @Test
    void rejectsNonQuantizedDtypes() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(128);
            assertThrows(IllegalArgumentException.class,
                    () -> TensorFactory.ofQuantized(source, DType.F32, 32));
        }
    }

    @Test
    void automaticBackendSelectionUsesCpuWhenMetalIsUnavailableAndDoesNotRequireHat() {
        Tensor tensor = TensorFactory.zeros(1);
        assertEquals("tech.kayys.aljabr.backend.cpu.CpuBackend", tensor.backend().getClass().getName());
    }

}
