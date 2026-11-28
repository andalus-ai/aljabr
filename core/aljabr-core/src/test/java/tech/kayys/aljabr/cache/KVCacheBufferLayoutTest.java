package tech.kayys.aljabr.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class KVCacheBufferLayoutTest {
    @Test
    void describesContiguousSequenceMajorKvGeometry() {
        KVCacheBufferLayout layout = KVCacheBufferLayout.sequenceMajor(12, 8, 64);
        assertEquals(12, layout.sequenceLength());
        assertEquals(8, layout.kvHeads());
        assertEquals(64, layout.headDimension());
        assertEquals(512, layout.sequenceStride());
        assertEquals(64, layout.headStride());
        assertEquals(1, layout.elementStride());
        assertEquals(6_144, layout.elementCount());
        assertEquals(24_576, layout.minimumByteLength(Float.BYTES));
    }

    @Test
    void calculatesMinimumSpanForStridedViews() {
        KVCacheBufferLayout layout = new KVCacheBufferLayout(2, 2, 2, 8, 3, 1);
        assertEquals(13L * Float.BYTES, layout.minimumByteLength(Float.BYTES));
    }

    @Test
    void rejectsOverlappingStridesAndInvalidElementWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheBufferLayout(2, 2, 4, 7, 4, 1));
        assertThrows(IllegalArgumentException.class,
                () -> KVCacheBufferLayout.sequenceMajor(1, 1, 1).minimumByteLength(0));
    }

    @Test
    void emptyVisiblePrefixRequiresNoBufferBytes() {
        KVCacheBufferLayout layout = KVCacheBufferLayout.sequenceMajor(0, 4, 32);
        assertEquals(0, layout.elementCount());
        assertEquals(0, layout.minimumByteLength(Float.BYTES));
    }
}
