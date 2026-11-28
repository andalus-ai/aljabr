package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HeapKVCacheBufferSharingTest {

    @Test
    void exportsCommittedPrefixAsFloat32HeapRegions() {
        KVCacheLayout layout = KVCacheLayout.uniform(2, 8, 2, 4);
        HeapKVCacheStorage storage = new HeapKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        float[] keys = new float[8];
        float[] values = new float[8];
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            session.write(layer, 0, keys, 0, values, 0);
        }
        session.commit(1);

        KVCacheBufferReadView view = session.openBufferReadView();
        try {
            assertEquals(1, view.visibleSequenceLength());
            assertEquals("cpu", view.backendId());
            assertEquals(KVCacheMemoryDomain.HOST_HEAP, view.keyRegion(0).memoryDomain());
            assertEquals(KVCacheElementType.FLOAT32, view.keyRegion(0).elementType());
            assertEquals(8L * Float.BYTES, view.keyRegion(0).byteLength());
            assertEquals(8, view.keyRegion(0).elementCount());
            assertInstanceOf(HeapKVCacheBufferHandle.class, view.keyRegion(0).handle());
            assertNotNull(view.readiness());
            assertDoesNotThrow(() -> view.readiness().toCompletableFuture().join());
        } finally {
            view.close();
        }
    }

    @Test
    void preventsDirectStorageMutationOfBorrowedPrefixAndAllowsAppend() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 4);
        HeapKVCacheStorage storage = new HeapKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        float[] vector = new float[4];
        session.write(0, 0, vector, 0, vector, 0);
        session.commit(1);

        KVCacheBufferReadView view = session.openBufferReadView();
        try {
            assertThrows(IllegalStateException.class,
                    () -> storage.write(0, 0, vector, 0, vector, 0));
            assertDoesNotThrow(() -> storage.write(0, 1, vector, 0, vector, 0));
            assertThrows(IllegalStateException.class, () -> storage.clearLayer(0));
            assertThrows(IllegalStateException.class, () -> storage.truncate(0));
        } finally {
            view.close();
        }

        assertDoesNotThrow(() -> storage.write(0, 0, vector, 0, vector, 0));
        assertDoesNotThrow(() -> storage.clearLayer(0));
    }

    @Test
    void viewCannotResolveRegionsAfterClose() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 4, 1, 4));
        KVCacheSession session = new KVCacheSession(storage);
        KVCacheBufferReadView view = session.openBufferReadView();
        view.close();

        assertThrows(IllegalStateException.class, () -> view.keyRegion(0));
        assertThrows(IllegalStateException.class, view::readiness);
    }

    @Test
    void nestedLeasesProtectLongestVisiblePrefixUntilAllClose() {
        KVCacheLayout layout = KVCacheLayout.uniform(1, 4, 1, 2);
        HeapKVCacheStorage storage = new HeapKVCacheStorage(layout);
        KVCacheSession session = new KVCacheSession(storage);
        float[] vector = new float[2];
        session.write(0, 0, vector, 0, vector, 0);
        session.commit(1);
        KVCacheBufferReadView first = session.openBufferReadView();
        session.write(0, 1, vector, 0, vector, 0);
        session.commit(2);
        KVCacheBufferReadView second = session.openBufferReadView();

        first.close();
        assertThrows(IllegalStateException.class,
                () -> storage.write(0, 1, vector, 0, vector, 0));
        second.close();
        assertDoesNotThrow(() -> storage.write(0, 1, vector, 0, vector, 0));
    }
    @Test
    void storageCannotCloseUntilLeaseIsReleased() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 4, 1, 4));
        KVCacheSession session = new KVCacheSession(storage);
        KVCacheBufferReadView view = session.openBufferReadView();

        assertThrows(IllegalStateException.class, storage::close);
        view.close();
        assertDoesNotThrow(storage::close);
        assertThrows(IllegalStateException.class, () -> session.openBufferReadView());
    }

    @Test
    void truncationAtVisibleBoundaryDoesNotInvalidateLease() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(
                KVCacheLayout.uniform(1, 4, 1, 2));
        KVCacheSession session = new KVCacheSession(storage);
        float[] vector = new float[2];
        session.write(0, 0, vector, 0, vector, 0);
        session.commit(1);
        KVCacheBufferReadView view = session.openBufferReadView();

        assertDoesNotThrow(() -> storage.truncate(1));
        assertEquals(1, view.visibleSequenceLength());
        assertThrows(IllegalStateException.class, () -> storage.truncate(0));
        view.close();
    }

}
