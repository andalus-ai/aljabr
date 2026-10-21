package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeNVFP4Block;

class GgufNVFP4SubTest {
    @Test
    void dequantizesNVFP4WithPerSubBlockScales() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(36);
            writeNVFP4Block(block, (byte) 0x40, (byte) 0x48, (byte) 0x38, (byte) 0, (byte) 0xA5);

            float[] row = new float[64];
            GgufBlockDequantizer.dequantizeNVFP4Block(block, 0, row, 0);

            assertEquals(6.0f, row[0], 0.0f);
            assertEquals(-2.0f, row[8], 0.0f);
            assertEquals(12.0f, row[16], 0.0f);
            assertEquals(-4.0f, row[24], 0.0f);
            assertEquals(3.0f, row[32], 0.0f);
            assertEquals(-1.0f, row[40], 0.0f);
            assertEquals(0.0f, row[48], 0.0f);
            assertEquals(0.0f, row[56], 0.0f);
        }
    }
}
