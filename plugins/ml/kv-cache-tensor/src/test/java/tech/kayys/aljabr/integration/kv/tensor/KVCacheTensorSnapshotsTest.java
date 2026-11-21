package tech.kayys.aljabr.integration.kv.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.cache.HeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.Tensor;

class KVCacheTensorSnapshotsTest {
    @Test
    void snapshotsCommittedKeysAndValuesIntoIndependentTensors() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 2, 2);
        HeapKVCacheStorage storage = new HeapKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        float[] keys = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] values = {11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f};
        session.writeRange(0, 0, 2, keys, 0, values, 0);
        session.commit(2);

        Tensor keyTensor = KVCacheTensorSnapshots.keys(session, 0, 0, 2, new CpuBackend());
        Tensor valueTensor = KVCacheTensorSnapshots.values(session, 0, 0, 2, new CpuBackend());
        try {
            assertArrayEquals(new long[] {2, 2, 2}, keyTensor.shape().dims());
            assertArrayEquals(keys, ((DefaultTensor) keyTensor).toFloatArray(), 0f);
            assertArrayEquals(values, ((DefaultTensor) valueTensor).toFloatArray(), 0f);

            session.reset();
            storage.close();

            // Snapshot tensors own their storage independently of the cache.
            assertArrayEquals(keys, ((DefaultTensor) keyTensor).toFloatArray(), 0f);
            assertArrayEquals(values, ((DefaultTensor) valueTensor).toFloatArray(), 0f);
        } finally {
            keyTensor.release();
            valueTensor.release();
        }
    }
}
