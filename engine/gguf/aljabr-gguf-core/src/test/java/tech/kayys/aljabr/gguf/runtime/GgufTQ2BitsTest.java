package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ2_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufTQFx.assertTQ2Pattern;

class GgufTQ2BitsTest {
    @Test
    void dequantizesTQ2_0PackedLanesInReferenceOrder() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(66);
            writeTQ2_0Block(block, (short) 0x3c00, (byte) 0x24);

            float[] row = new float[256];
            GgufBlockDequantizer.dequantizeTQ2_0Block(block, 0, row, 0);

            assertTQ2Pattern(row, -1.0f, 0.0f, 1.0f, -1.0f);
        }
    }
}
