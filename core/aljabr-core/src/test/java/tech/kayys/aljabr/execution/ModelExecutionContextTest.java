package tech.kayys.aljabr.execution;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.HeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheStorage;

import static org.junit.jupiter.api.Assertions.*;

class ModelExecutionContextTest {

    @Test
    void exposesCallerSuppliedStorageWithoutTakingOwnership() {
        KVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(2, 8, 1, 4));

        ModelExecutionContext context =
                ModelExecutionContext.withKVCacheStorage("model-a", "request-a", storage);

        assertEquals("model-a", context.modelId());
        assertEquals("request-a", context.requestId());
        assertSame(storage, context.kvCacheStorage().orElseThrow());

        // The context is only a binding; storage remains usable and caller-owned.
        float[] key = {1, 2, 3, 4};
        float[] value = {5, 6, 7, 8};
        storage.write(0, 0, key, 0, value, 0);
        float[] actual = new float[4];
        storage.readKey(0, 0, actual, 0);
        assertArrayEquals(key, actual);

        storage.close();
    }

    @Test
    void allowsRuntimesToKeepTheirOwnCacheState() {
        ModelExecutionContext context =
                ModelExecutionContext.withoutKVCache("model-native", "request-native");

        assertTrue(context.kvCacheStorage().isEmpty());
    }

    @Test
    void rejectsMissingExecutionIdentity() {
        assertThrows(IllegalArgumentException.class,
                () -> ModelExecutionContext.withoutKVCache(" ", "request"));
        assertThrows(IllegalArgumentException.class,
                () -> ModelExecutionContext.withoutKVCache("model", null));
    }

    @Test
    void requiresStorageWhenExplicitlyBindingIt() {
        assertThrows(NullPointerException.class,
                () -> ModelExecutionContext.withKVCacheStorage("model", "request", null));
    }
}
