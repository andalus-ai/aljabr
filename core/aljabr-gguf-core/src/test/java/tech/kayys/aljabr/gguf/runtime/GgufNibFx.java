package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Shared nibble-lane fixture helpers for compact 32-value quantized blocks.
 */
final class GgufNibFx {
    private GgufNibFx() {
    }

    static void writeLaneOrder(MemorySegment block, long offset) {
        for (int i = 0; i < 16; i++) {
            block.set(ValueLayout.JAVA_BYTE, offset + i, (byte) (low(i) | (high(i) << 4)));
        }
    }

    static int low(int index) {
        return index & 0x0F;
    }

    static int high(int index) {
        return 15 - index;
    }
}
