package tech.kayys.aljabr.cache;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tech.kayys.aljabr.cache.testing.KVCacheBufferProviderContractTest;

class NativeKVCacheBufferProviderContractTest extends KVCacheBufferProviderContractTest {
    @Override
    protected Provider openProvider() {
        NativeKVCacheStorage storage = new NativeKVCacheStorage(
                new KVCacheLayout(5, new int[] {1, 2}, new int[] {2, 3}));
        return new Provider(storage, "cpu", KVCacheMemoryDomain.HOST_NATIVE,
                (lease, region) -> NativeKVCacheBufferAccess.withReadOnlySegment(
                        lease, region, segment -> segment.toArray(ValueLayout.JAVA_FLOAT)));
    }
}
