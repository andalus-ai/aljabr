package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

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
    void rejectsNegativeLayer() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheRouting.resolveCacheLayer(-1, false, false));
    }
}
