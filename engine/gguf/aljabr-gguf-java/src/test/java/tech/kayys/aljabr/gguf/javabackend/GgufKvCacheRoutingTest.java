package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufKvCacheRoutingTest {

    @Test
    void layerWithKeyProjectionOwnsItsCache() {
        GGUFTensorInfo[] keys = keyWeights(false, false, false, false, true);
        assertEquals(4, GgufKvCacheRouting.resolveCacheLayer(4, keys, false));
        assertEquals(4, GgufKvCacheRouting.resolveCacheLayer(4, keys, true));
    }

    @Test
    void sharedGlobalLayerUsesImmediatelyPrecedingLayerWhenItOwnsKv() {
        GGUFTensorInfo[] keys = keyWeights(false, false, true, true, false);
        assertEquals(3, GgufKvCacheRouting.resolveCacheLayer(4, keys, false));
    }

    @Test
    void sharedSlidingWindowLayerUsesTwoBackOwnerWhenItOwnsKv() {
        GGUFTensorInfo[] keys = keyWeights(false, true, true, false, false);
        assertEquals(2, GgufKvCacheRouting.resolveCacheLayer(4, keys, true));
    }

    @Test
    void rejectsCalculatedOwnerWithoutKeyProjection() {
        GGUFTensorInfo[] keys = keyWeights(false, false, false, false, false);
        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheRouting.resolveCacheLayer(4, keys, false));
    }

    @Test
    void rejectsLeadingSharedLayerWithoutAnyOwner() {
        GGUFTensorInfo[] keys = keyWeights(false, true);
        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheRouting.resolveCacheLayer(0, keys, false));
    }

    @Test
    void leadingSharedLayerCanUseLayerZeroWhenItOwnsKv() {
        GGUFTensorInfo[] keys = keyWeights(true, false);
        assertEquals(0, GgufKvCacheRouting.resolveCacheLayer(1, keys, true));
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
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheRouting.validateCacheGeometry(cache, 1, 4, 64));
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
                () -> GgufKvCacheRouting.resolveCacheLayer(-1, keyWeights(true), false));
    }

    private static GGUFTensorInfo[] keyWeights(boolean... hasKeyProjection) {
        GGUFTensorInfo[] weights = new GGUFTensorInfo[hasKeyProjection.length];
        for (int layer = 0; layer < hasKeyProjection.length; layer++) {
            if (hasKeyProjection[layer]) {
                weights[layer] = new GGUFTensorInfo("blk." + layer + ".attn_k.weight",
                        new long[] {1}, 0, 0L, Float.BYTES);
            }
        }
        return weights;
    }
}