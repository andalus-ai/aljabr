package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ1_0Block;

class GgufQ1BitsTest {
    @Test
    void dequantizesQ1_0BitsLeastSignificantBitFirst() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(18);
            writeQ1_0Block(block, (short) 0x3c00, (byte) 0x01);

            float[] row = new float[128];
            GgufBlockDequantizer.dequantizeQ1_0Block(block, 0, row, 0);

            assertEquals(1.0f, row[0], 0.0f);
            assertEquals(-1.0f, row[1], 0.0f);
            assertEquals(-1.0f, row[7], 0.0f);
        }
    }
}
