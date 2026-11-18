package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KVCacheLayoutTest {

    @Test
    void supportsUniformLayerGeometry() {
        KVCacheLayout layout = KVCacheLayout.uniform(4, 128, 8, 64);

        assertEquals(4, layout.layerCount());
        assertEquals(128, layout.maxSequenceLength());
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            assertEquals(8, layout.kvHeads(layer));
            assertEquals(64, layout.headDimension(layer));
            assertEquals(512, layout.layerStride(layer));
        }
    }

    @Test
    void supportsHybridPerLayerGeometry() {
        KVCacheLayout layout = new KVCacheLayout(
                4096,
                new int[] {1, 1, 8, 8},
                new int[] {256, 256, 512, 512});

        assertEquals(256, layout.layerStride(0));
        assertEquals(256, layout.layerStride(1));
        assertEquals(4096, layout.layerStride(2));
        assertEquals(4096, layout.layerStride(3));
    }

    @Test
    void defensivelyCopiesGeometryArrays() {
        int[] heads = {2, 4};
        int[] dimensions = {64, 128};
        KVCacheLayout layout = new KVCacheLayout(1024, heads, dimensions);

        heads[0] = 99;
        dimensions[0] = 99;
        layout.kvHeadsPerLayer()[1] = 99;
        layout.headDimensionsPerLayer()[1] = 99;

        assertEquals(2, layout.kvHeads(0));
        assertEquals(64, layout.headDimension(0));
        assertEquals(4, layout.kvHeads(1));
        assertEquals(128, layout.headDimension(1));
    }

    @Test
    void rejectsInvalidLayerGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheLayout(32, new int[] {2}, new int[] {0}));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheLayout(32, new int[] {2}, new int[] {64, 128}));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheLayout(32, new int[] {Integer.MAX_VALUE},
                        new int[] {Integer.MAX_VALUE}));
    }

    @Test
    void rejectsInvalidLayerAccess() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 32, 2, 64);

        assertThrows(IndexOutOfBoundsException.class, () -> layout.kvHeads(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> layout.headDimension(2));
    }
}
