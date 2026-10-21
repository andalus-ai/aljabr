package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.memory.OffHeapBufferPool;
import tech.kayys.aljabr.backend.cpu.quantize.QuantizeOps;
import tech.kayys.aljabr.backend.cpu.quantize.DequantizeOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuPhase34Test {

    @Test
    void testBufferPoolRecyclesMemory() {
        try (OffHeapBufferPool pool = new OffHeapBufferPool()) {
            MemorySegment seg1 = pool.acquire(1024);
            assertEquals(1024, seg1.byteSize());

            pool.release(seg1);

            MemorySegment seg2 = pool.acquire(1024);
            assertEquals(1024, seg2.byteSize());

            // Should be the exact same underlying memory address because it was recycled
            assertEquals(seg1.address(), seg2.address());
        }
    }

    @Test
    void testQuantizeDequantizeQ8_0() {
        try (Arena arena = Arena.ofConfined()) {
            int numElems = QuantizeOps.Q8_0_BLOCK_ELEMS * 2; // 2 blocks
            MemorySegment src = arena.allocate(numElems * 4L);
            MemorySegment qDst = arena.allocate(QuantizeOps.Q8_0_BLOCK_BYTES * 2L);
            MemorySegment out = arena.allocate(numElems * 4L);

            // Populate src with some values
            for (int i = 0; i < numElems; i++) {
                src.set(ValueLayout.JAVA_FLOAT, i * 4L, (float) Math.sin(i));
            }

            // Quantize
            QuantizeOps.quantizeQ8_0(src, qDst, numElems);

            // Dequantize
            DequantizeOps.dequantizeQ8_0(qDst, out, numElems);

            // Verify accuracy within tolerance
            for (int i = 0; i < numElems; i++) {
                float expected = src.get(ValueLayout.JAVA_FLOAT, i * 4L);
                float actual = out.get(ValueLayout.JAVA_FLOAT, i * 4L);

                // Q8_0 has 8-bit precision, so tolerance is roughly ~ 1/127 of max abs val
                assertTrue(Math.abs(expected - actual) < 0.05f,
                        "Mismatch at index " + i + ": expected " + expected + ", got " + actual);
            }
        }
    }
}
