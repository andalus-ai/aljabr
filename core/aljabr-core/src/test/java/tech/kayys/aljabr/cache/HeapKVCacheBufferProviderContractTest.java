package tech.kayys.aljabr.cache;

import java.nio.FloatBuffer;

import tech.kayys.aljabr.cache.testing.KVCacheBufferProviderContractTest;

class HeapKVCacheBufferProviderContractTest extends KVCacheBufferProviderContractTest {
    @Override
    protected Provider openProvider() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                new KVCacheLayout(5, new int[] {1, 2}, new int[] {2, 3}));
        return new Provider(storage, "cpu", KVCacheMemoryDomain.HOST_HEAP,
                (lease, region) -> HeapKVCacheBufferAccess.withReadOnlyBuffer(
                        lease, region, HeapKVCacheBufferProviderContractTest::read));
    }

    private static float[] read(FloatBuffer buffer) {
        float[] values = new float[buffer.remaining()];
        buffer.get(values);
        return values;
    }
}
