package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ1_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufTQFx.assertTQ1Pattern;

class GgufTQ1BitsTest {
    @Test
    void dequantizesTQ1_0PackedTritsInReferenceOrder() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment block = arena.allocate(54);
            writeTQ1_0Block(block, (short) 0x3c00, (byte) 49);

            float[] row = new float[256];
            GgufBlockDequantizer.dequantizeTQ1_0Block(block, 0, row, 0);

            assertTQ1Pattern(row, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f);
        }
    }
}
