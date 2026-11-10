package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufPrefillMetalCacheSyncTest {
    @Test
    void layerWithOwnKeyProjectionUsesItsOwnHostCache() {
        assertEquals(4, GgufPrefillMetalCacheSync.sourceLayer(4, true, false));
    }

    @Test
    void globalAttentionSharedKvUsesPreviousLayer() {
        assertEquals(3, GgufPrefillMetalCacheSync.sourceLayer(4, false, false));
    }

    @Test
    void swaSharedKvUsesTwoLayersBack() {
        assertEquals(2, GgufPrefillMetalCacheSync.sourceLayer(4, false, true));
    }

    @Test
    void sharedKvSourceNeverBecomesNegative() {
        assertEquals(0, GgufPrefillMetalCacheSync.sourceLayer(0, false, true));
        assertEquals(0, GgufPrefillMetalCacheSync.sourceLayer(1, false, true));
    }
}
