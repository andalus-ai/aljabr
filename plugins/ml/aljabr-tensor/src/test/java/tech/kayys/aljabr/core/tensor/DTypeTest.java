package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DTypeTest {

    @Test
    void computesPrimitiveStorageFootprints() {
        assertEquals(0L, DType.F32.memoryFootprintBytes(0));
        assertEquals(16L, DType.F32.memoryFootprintBytes(4));
        assertEquals(6L, DType.F16.memoryFootprintBytes(3));
        assertEquals(6L, DType.BF16.memoryFootprintBytes(3));
        assertEquals(12L, DType.I32.memoryFootprintBytes(3));
        assertEquals(3L, DType.I8.memoryFootprintBytes(3));
        assertEquals(3L, DType.INT8.memoryFootprintBytes(3));
    }

    @Test
    void computesBlockQuantizedStorageFootprints() {
        assertEquals(34L, DType.Q8_0.memoryFootprintBytes(32));
        assertEquals(68L, DType.Q8_0.memoryFootprintBytes(64));
        assertEquals(18L, DType.Q4_0.memoryFootprintBytes(32));
        assertEquals(144L, DType.Q4_K.memoryFootprintBytes(256));
        assertEquals(0L, DType.Q4_K.memoryFootprintBytes(0));
    }

    @Test
    void rejectsNegativeElementCounts() {
        for (DType dtype : DType.values()) {
            IllegalArgumentException error = assertThrows(
                    IllegalArgumentException.class,
                    () -> dtype.memoryFootprintBytes(-1),
                    dtype + " should reject negative element counts");
            assertTrue(error.getMessage().contains("non-negative"));
        }
    }

    @Test
    void rejectsCountsThatDoNotAlignToQuantizationBlocks() {
        assertThrows(IllegalArgumentException.class, () -> DType.Q8_0.memoryFootprintBytes(31));
        assertThrows(IllegalArgumentException.class, () -> DType.Q4_0.memoryFootprintBytes(33));
        assertThrows(IllegalArgumentException.class, () -> DType.Q4_K.memoryFootprintBytes(255));
    }

    @Test
    void rejectsStorageFootprintOverflow() {
        assertThrows(IllegalArgumentException.class, () -> DType.F32.memoryFootprintBytes(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> DType.Q8_0.memoryFootprintBytes(Long.MAX_VALUE - 31));
    }
}
