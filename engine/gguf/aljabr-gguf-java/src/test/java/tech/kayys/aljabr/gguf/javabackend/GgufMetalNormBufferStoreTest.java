package tech.kayys.aljabr.gguf.javabackend;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufMetalNormBufferStoreTest {
    @Test
    void registersModelAndHeadNormalizationSlotsAsLayerSlices() {
        GgufMetalNormBufferStore store = new GgufMetalNormBufferStore();
        float[][] postAttention = {{1f, 2f, 3f}, {4f, 5f, 6f}};
        float[][] ffn = {{7f, 8f, 9f}, {10f, 11f, 12f}};
        float[][] queryNorm = {{13f, 14f}, {15f, 16f}};

        MemorySegment[] slots = store.register(
                2, 3, 2, postAttention, ffn, null, null, queryNorm, null);

        assertEquals(6, slots.length);
        assertEquals(1f, value(store.slice(0, 0), 0), 0f);
        assertEquals(5f, value(store.slice(0, 1), 1), 0f);
        assertEquals(8f, value(store.slice(1, 0), 1), 0f);
        assertEquals(15f, value(store.slice(4, 1), 0), 0f);
        assertEquals(16f, value(store.slice(4, 1), 1), 0f);
        assertEquals(0f, value(store.slice(2, 0), 0), 0f);
        assertEquals(0f, value(store.slice(5, 1), 1), 0f);
        assertEquals(MemorySegment.NULL, store.slice(6, 0));
        assertEquals(MemorySegment.NULL, store.slice(0, 2));
        assertEquals(2L * 3 * Float.BYTES, store.slotByteSizes(2, 3, 2)[0]);
        assertEquals(2L * 2 * Float.BYTES, store.slotByteSizes(2, 3, 2)[4]);
    }

    @Test
    void replacingRegistrationPublishesTheNewGeneration() {
        GgufMetalNormBufferStore store = new GgufMetalNormBufferStore();
        store.register(1, 2, 2, new float[][] {{1f, 2f}}, null, null, null, null, null);
        store.register(1, 2, 2, new float[][] {{8f, 9f}}, null, null, null, null, null);

        assertEquals(8f, value(store.slice(0, 0), 0), 0f);
        assertEquals(9f, value(store.slice(0, 0), 1), 0f);
        assertEquals(2L * Float.BYTES, store.slice(0, 0).byteSize());
    }

    private static float value(MemorySegment segment, long index) {
        return segment.get(ValueLayout.JAVA_FLOAT, index * Float.BYTES);
    }
}
