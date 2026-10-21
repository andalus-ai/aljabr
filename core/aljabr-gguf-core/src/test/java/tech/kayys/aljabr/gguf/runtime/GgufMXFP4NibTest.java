package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeMXFP4Block;

class GgufMXFP4NibTest {
    @Test
    void dequantizesMXFP4NibblesIntoSplitHalves() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(17);
            writeMXFP4Block(block, (byte) 128, (byte) 0xA5);

            float[] row = new float[32];
            GgufBlockDequantizer.dequantizeMXFP4Block(block, 0, row, 0);

            assertEquals(6.0f, row[0], 0.0f);
            assertEquals(-2.0f, row[16], 0.0f);
        }
    }
}
