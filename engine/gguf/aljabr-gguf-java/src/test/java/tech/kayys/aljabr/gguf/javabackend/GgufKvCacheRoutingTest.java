package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufKvCacheRoutingTest {

    @Test
    void layerWithKeyProjectionOwnsItsCache() {
        assertEquals(4, GgufKvCacheRouting.resolveCacheLayer(4, true, false));
        assertEquals(4, GgufKvCacheRouting.resolveCacheLayer(4, true, true));
    }

    @Test
    void sharedGlobalLayerUsesImmediatelyPrecedingLayer() {
        assertEquals(3, GgufKvCacheRouting.resolveCacheLayer(4, false, false));
    }

    @Test
    void sharedSlidingWindowLayerUsesLayerTwoBack() {
        assertEquals(2, GgufKvCacheRouting.resolveCacheLayer(4, false, true));
    }

    @Test
    void sharedLeadingLayersClampOwnerToLayerZero() {
        assertEquals(0, GgufKvCacheRouting.resolveCacheLayer(0, false, false));
        assertEquals(0, GgufKvCacheRouting.resolveCacheLayer(1, false, true));
    }

    @Test
    void metalCacheSynchronizationUsesTheCanonicalRoutingPolicy() {
        assertEquals(4, GgufKvCacheRouting.resolveCacheLayer(4, true, false));
        assertEquals(3, GgufKvCacheRouting.resolveCacheLayer(4, false, false));
        assertEquals(2, GgufKvCacheRouting.resolveCacheLayer(4, false, true));
        assertEquals(0, GgufKvCacheRouting.resolveCacheLayer(1, false, true));
    }

    @Test
    void acceptsMatchingOwnerGeometry() {
        KVCache cache = new KVCache(3, 16,
                new int[] {2, 2, 4}, new int[] {64, 64, 128});

        GgufKvCacheRouting.validateCacheGeometry(cache, 2, 4, 128);
    }

    @Test
    void rejectsOwnerWithDifferentKvHeadCount() {
        KVCache cache = new KVCache(3, 16,
                new int[] {2, 2, 4}, new int[] {64, 64, 128});

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheRouting.validateCacheGeometry(cache, 1, 4, 64));
        org.junit.jupiter.api.Assertions.assertTrue(error.getMessage().contains("geometry mismatch"));
    }

    @Test
    void rejectsOwnerWithDifferentHeadDimension() {
        KVCache cache = new KVCache(3, 16,
                new int[] {2, 2, 4}, new int[] {64, 64, 128});

        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheRouting.validateCacheGeometry(cache, 0, 2, 128));
    }

    @Test
    void rejectsNegativeLayer() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheRouting.resolveCacheLayer(-1, false, false));
    }
}
