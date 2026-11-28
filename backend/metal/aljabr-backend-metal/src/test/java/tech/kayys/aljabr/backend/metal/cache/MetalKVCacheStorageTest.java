package tech.kayys.aljabr.backend.metal.cache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.cache.KVCacheBufferReadView;
import tech.kayys.aljabr.cache.KVCacheBufferRegion;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.metal.binding.MetalBinding;

class MetalKVCacheStorageTest {
    private static Optional<MetalBinding> availableBinding() {
        if (!MetalBinding.initialize()) return Optional.empty();
        MetalBinding binding = MetalBinding.getInstance();
        binding.init();
        return binding.supportsKVBufferAllocation() ? Optional.of(binding) : Optional.empty();
    }

    @Test
    void sharedMetalBuffersExposeBoundedBorrowedRegions() {
        MetalBinding binding = availableBinding().orElse(null);
        Assumptions.assumeTrue(binding != null, "Requires an available unified-memory Metal device");
        KVCacheLayout layout = KVCacheLayout.uniform(1, 8, 1, 2);
        try (MetalKVCacheStorage storage = new MetalKVCacheStorage(layout, binding)) {
            KVCacheSession session = new KVCacheSession(storage);
            session.write(0, 0, new float[] {1, 2}, 0, new float[] {3, 4}, 0);
            session.commit(1);
            try (KVCacheBufferReadView lease = session.openBufferReadView()) {
                KVCacheBufferRegion region = lease.keyRegion(0);
                assertEquals("metal", region.backendId());
                assertEquals(2, region.elementCount());
                float[] values = MetalKVCacheBufferAccess.withBuffer(lease, region, view ->
                        new float[] {
                                view.readOnlyContents().get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0),
                                view.readOnlyContents().get(java.lang.foreign.ValueLayout.JAVA_FLOAT, Float.BYTES)
                        });
                assertArrayEquals(new float[] {1, 2}, values);
                assertThrows(UnsupportedOperationException.class, () ->
                        MetalKVCacheBufferAccess.withBuffer(lease, region, view -> {
                            view.readOnlyContents().set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 9);
                            return null;
                        }));
            }
        }
    }

    @Test
    void invalidatesBorrowedSegmentAfterCallbackReturns() {
        MetalBinding binding = availableBinding().orElse(null);
        Assumptions.assumeTrue(binding != null, "Requires an available unified-memory Metal device");
        try (MetalKVCacheStorage storage = new MetalKVCacheStorage(
                KVCacheLayout.uniform(1, 4, 1, 2), binding)) {
            KVCacheSession session = new KVCacheSession(storage);
            KVCacheBufferReadView lease = session.openBufferReadView();
            KVCacheBufferRegion region = lease.keyRegion(0);
            AtomicReference<MemorySegment> escaped = new AtomicReference<>();
            MetalKVCacheBufferAccess.withBuffer(lease, region, view -> {
                escaped.set(view.readOnlyContents());
                return null;
            });
            assertThrows(IllegalStateException.class, () -> escaped.get().byteSize());
            lease.close();
        }
    }

    @Test
    void protectsBorrowedPrefixAndRejectsCloseUntilLeaseReleased() {
        MetalBinding binding = availableBinding().orElse(null);
        Assumptions.assumeTrue(binding != null, "Requires an available unified-memory Metal device");
        MetalKVCacheStorage storage = new MetalKVCacheStorage(
                KVCacheLayout.uniform(1, 8, 1, 2), binding);
        KVCacheSession session = new KVCacheSession(storage);
        session.write(0, 0, new float[] {1, 2}, 0, new float[] {3, 4}, 0);
        session.commit(1);
        KVCacheBufferReadView lease = session.openBufferReadView();
        assertThrows(IllegalStateException.class, () ->
                storage.write(0, 0, new float[] {5, 6}, 0, new float[] {7, 8}, 0));
        storage.write(0, 1, new float[] {5, 6}, 0, new float[] {7, 8}, 0);
        assertThrows(IllegalStateException.class, storage::close);
        lease.close();
        storage.close();
        assertThrows(IllegalStateException.class, () -> storage.readKey(0, 0, new float[2], 0));
    }
}
