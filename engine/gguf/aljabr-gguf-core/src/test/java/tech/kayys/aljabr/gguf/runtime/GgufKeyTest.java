package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class GgufKeyTest {
    @Test
    void validatedShapeKeyMatchesDerivedTensorKey() {
        GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 3}, 12, 64, 432);

        assertEquals(GgufKey.from(tensor), GgufKey.from(tensor, 256, 3));
        assertEquals(GgufKey.from(tensor).hashCode(), GgufKey.from(tensor, 256, 3).hashCode());
    }

    @Test
    void reusesCachedKeyForRepeatedTensorLookups() {
        GgufKey.clearCaches();
        assertEquals(0, GgufKey.recentKeyFastCacheSize());
        GGUFTensorInfo tensor = new GGUFTensorInfo("q4.cache", new long[]{256, 3}, 12, 64, 432);

        try {
            GgufKey first = GgufKey.from(tensor);
            assertEquals(1, GgufKey.recentKeyFastCacheSize());

            assertSame(first, GgufKey.from(tensor));
            assertSame(first, GgufKey.from(tensor, 256, 3));
            assertEquals(1, GgufKey.recentKeyFastCacheSize());
        } finally {
            GgufKey.clearCaches();
        }
    }

    @Test
    void keepsExplicitDimensionKeysDistinctFromShapeKey() {
        GgufKey.clearCaches();
        GGUFTensorInfo tensor = new GGUFTensorInfo("q4.explicit", new long[]{256, 3}, 12, 64, 432);

        try {
            GgufKey shapeKey = GgufKey.from(tensor);
            GgufKey explicitKey = GgufKey.from(tensor, 128, 6);

            assertNotEquals(shapeKey, explicitKey);
            assertSame(explicitKey, GgufKey.from(tensor, 128, 6));
            assertSame(shapeKey, GgufKey.from(tensor));
            assertEquals(1, GgufKey.recentKeyFastCacheSize());
        } finally {
            GgufKey.clearCaches();
        }
    }

    @Test
    void reusesRecentKeysAcrossAlternatingTensorLookups() {
        GGUFTensorInfo first = new GGUFTensorInfo("q4.first", new long[]{256, 3}, 12, 64, 432);
        GGUFTensorInfo second = new GGUFTensorInfo("q4.second", new long[]{256, 4}, 12, 512, 576);

        GgufKey firstKey = GgufKey.from(first);
        GgufKey secondKey = GgufKey.from(second);

        assertSame(firstKey, GgufKey.from(first));
        assertSame(secondKey, GgufKey.from(second));
    }
}
