package tech.kayys.aljabr.integration.kv;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.cache.HeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;

class KVCacheTensorBridgeTest {
    @Test
    void snapshotCopiesCommittedLayerRangeIntoIndependentTensors() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 8, 2, 2);
        HeapKVCacheStorage storage = new HeapKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        CpuBackend backend = new CpuBackend();
        float[] keys = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
        float[] values = { 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f };
        session.writeRange(0, 0, 2, keys, 0, values, 0);
        session.commit(2);

        try (KVCacheTensorSnapshot snapshot = KVCacheTensorBridge.snapshot(
                session, 0, 0, 2, backend)) {
            assertEquals(0, snapshot.layer());
            assertEquals(0, snapshot.firstPosition());
            assertEquals(2, snapshot.tokenCount());
            assertArrayEquals(new long[] { 2, 2, 2 }, snapshot.keys().shape().dims());
            assertArrayEquals(keys, snapshot.keys().toFloatArray());
            assertArrayEquals(values, snapshot.values().toFloatArray());

            storage.clearLayer(0);
            assertArrayEquals(keys, snapshot.keys().toFloatArray());
            assertArrayEquals(values, snapshot.values().toFloatArray());
        } finally {
            storage.close();
            backend.close();
        }
    }

    @Test
    void snapshotRejectsRangesOutsideCommittedPrefix() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 4, 1, 2);
        HeapKVCacheStorage storage = new HeapKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        CpuBackend backend = new CpuBackend();
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> KVCacheTensorBridge.snapshot(session, 0, 0, 1, backend));
        } finally {
            storage.close();
            backend.close();
        }
    }
}
