package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Shared little-endian half-word writers for F16/BF16 GGUF runtime tests.
 */
final class GgufHalfFx {
    static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufHalfFx() {
    }

    static void writeShorts(MemorySegment segment, short[] values) {
        for (int i = 0; i < values.length; i++) {
            segment.set(LE_SHORT, i * (long) Short.BYTES, values[i]);
        }
    }
}
