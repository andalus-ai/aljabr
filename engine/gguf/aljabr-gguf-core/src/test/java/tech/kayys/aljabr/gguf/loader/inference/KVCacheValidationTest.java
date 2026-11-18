package tech.kayys.aljabr.gguf.loader.inference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class KVCacheValidationTest {

    @Test
    void constructorsRejectInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new KVCache(0, 4, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new KVCache(1, 0, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new KVCache(1, 4, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> new KVCache(1, 4, 1, -2));
        assertThrows(IllegalArgumentException.class, () -> new KVCache(2, 4, new int[] {1}, new int[] {2}));
        assertThrows(IllegalArgumentException.class, () -> new KVCache(1, 4, new int[] {1}, new int[] {0}));
    }

    @Test
    void advanceCannotMoveBeyondCapacityOrBackwards() {
        KVCache cache = new KVCache(1, 3, 1, 2);

        cache.advance(2);
        assertEquals(2, cache.seqLen());
        assertThrows(IllegalArgumentException.class, () -> cache.advance(2));
        assertThrows(IllegalArgumentException.class, () -> cache.advance(-1));
        assertEquals(2, cache.seqLen());
    }

    @Test
    void storeRejectsCapacityOverflowAndShortVectors() {
        KVCache cache = new KVCache(1, 1, 1, 2);
        assertThrows(IllegalArgumentException.class,
                () -> cache.store(0, new float[] {1}, new float[] {2, 3}));
        cache.store(0, new float[] {1, 2}, new float[] {3, 4});
        cache.advance();
        assertThrows(IllegalStateException.class,
                () -> cache.store(0, new float[] {5, 6}, new float[] {7, 8}));
    }

    @Test
    void explicitAndBatchWritesRejectOutOfRangeWrites() {
        KVCache cache = new KVCache(1, 2, 1, 2);
        assertThrows(IndexOutOfBoundsException.class,
                () -> cache.storeDirect(0, 2, new float[] {1, 2}, new float[] {3, 4}, 2));
        assertThrows(IllegalArgumentException.class,
                () -> cache.storeDirect(0, 0, new float[] {1}, new float[] {2}, 2));
        assertThrows(IndexOutOfBoundsException.class,
                () -> cache.storeBatch(0, 1, new float[] {1, 2, 3, 4},
                        0, new float[] {1, 2, 3, 4}, 0, 4));
    }

    @Test
    void copyFromRejectsIncompatibleLayerGeometry() {
        KVCache source = new KVCache(1, 4, 2, 2);
        source.advance();
        KVCache destination = new KVCache(1, 4, 1, 4);

        assertThrows(IllegalArgumentException.class, () -> destination.copyFrom(source, 1));
        assertEquals(0, destination.seqLen());
    }
}
