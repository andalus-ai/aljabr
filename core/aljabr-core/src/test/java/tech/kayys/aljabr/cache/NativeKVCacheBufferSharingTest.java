package tech.kayys.aljabr.cache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class NativeKVCacheBufferSharingTest {
    @Test
    void exportsNativeReadOnlyRegionsAndPreservesValues() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 8, 1, 2);
        try (NativeKVCacheStorage storage = new NativeKVCacheStorage(layout)) {
            KVCacheSession session = new KVCacheSession(storage);
            session.write(0, 0, new float[] {1, 2}, 0, new float[] {3, 4}, 0);
            session.commit(1);
            try (KVCacheBufferReadView lease = session.openBufferReadView()) {
                KVCacheBufferRegion key = lease.keyRegion(0);
                KVCacheBufferRegion value = lease.valueRegion(0);
                assertEquals(KVCacheMemoryDomain.HOST_NATIVE, key.memoryDomain());
                assertEquals("cpu", key.backendId());
                assertEquals(KVCacheElementType.FLOAT32, key.elementType());
                assertEquals(2, key.elementCount());
                assertEquals(2, value.elementCount());
                float[] observed = NativeKVCacheBufferAccess.withReadOnlySegment(lease, key, segment ->
                        new float[] {segment.get(ValueLayout.JAVA_FLOAT, 0),
                                segment.get(ValueLayout.JAVA_FLOAT, Float.BYTES)});
                assertArrayEquals(new float[] {1, 2}, observed);
                assertThrows(UnsupportedOperationException.class, () ->
                        NativeKVCacheBufferAccess.withReadOnlySegment(lease, key, segment -> {
                            segment.set(ValueLayout.JAVA_FLOAT, 0, 99);
                            return null;
                        }));
            }
            float[] key = new float[2];
            float[] value = new float[2];
            session.readKey(0, 0, key, 0);
            session.readValue(0, 0, value, 0);
            assertArrayEquals(new float[] {1, 2}, key);
            assertArrayEquals(new float[] {3, 4}, value);
        }
    }

    @Test
    void protectsBorrowedPrefixAllowsAppendAndClosesSafely() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 2);
        NativeKVCacheStorage storage = new NativeKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        session.write(0, 0, new float[] {1, 2}, 0, new float[] {3, 4}, 0);
        session.commit(1);
        KVCacheBufferReadView lease = session.openBufferReadView();
        assertThrows(IllegalStateException.class, () ->
                storage.write(0, 0, new float[] {5, 6}, 0, new float[] {7, 8}, 0));
        storage.write(0, 1, new float[] {5, 6}, 0, new float[] {7, 8}, 0);
        assertThrows(IllegalStateException.class, storage::close);
        lease.close();
        lease.close();
        storage.write(0, 0, new float[] {9, 10}, 0, new float[] {11, 12}, 0);
        storage.close();
        assertThrows(IllegalStateException.class, () -> storage.readKey(0, 0, new float[2], 0));
    }

    @Test
    void rejectsTruncationThroughBorrowedPrefixAndAllowsBoundary() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 2);
        try (NativeKVCacheStorage storage = new NativeKVCacheStorage(layout)) {
            KVCacheSession session = new KVCacheSession(storage);
            session.writeRange(0, 0, 2, new float[] {1, 2, 3, 4}, 0, new float[] {5, 6, 7, 8}, 0);
            session.commit(2);
            try (KVCacheBufferReadView lease = session.openBufferReadView()) {
                assertThrows(IllegalStateException.class, () -> storage.truncate(1));
                storage.truncate(2);
            }
        }
    }

    @Test
    void rejectsForeignRegionAccess() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 4, 1, 2);
        try (NativeKVCacheStorage first = new NativeKVCacheStorage(layout);
                NativeKVCacheStorage second = new NativeKVCacheStorage(layout)) {
            KVCacheSession firstSession = new KVCacheSession(first);
            KVCacheSession secondSession = new KVCacheSession(second);
            try (KVCacheBufferReadView firstLease = firstSession.openBufferReadView();
                    KVCacheBufferReadView secondLease = secondSession.openBufferReadView()) {
                KVCacheBufferRegion foreign = secondLease.keyRegion(0);
                assertThrows(IllegalStateException.class, () ->
                        NativeKVCacheBufferAccess.withReadOnlySegment(firstLease, foreign, MemorySegment::byteSize));
            }
        }
    }
}
