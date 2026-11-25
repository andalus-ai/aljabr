package tech.kayys.aljabr.core.memory.offheap;

import org.junit.jupiter.api.Test;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.lang.foreign.Arena;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class OffHeapMemoryStoreTest {

    @Test
    void testZeroCopyGet {
        try (OffHeapMemoryStore store = new OffHeapMemoryStore()) {
            byte[] key = "test-key-offheap".getBytes();

            try (Arena arena = Arena.ofConfined()) {
                // Prepare some data
                MemorySegment input = arena.allocate(150);
                for (int i = 0; i < 150; i++) {
                    input.set(ValueLayout.JAVA_BYTE, i, (byte) (i * 2));
                }

                // Put
                store.put(key, input);

                // Get Zero Copy
                Optional<MemorySegment> outputOpt = store.getZeroCopy(key, 150, arena);
                assertTrue(outputOpt.isPresent());

                MemorySegment output = outputOpt.get();
                assertEquals(150, output.byteSize());

                // Verify byte values — this should be a pointer into the store's storage,
                // not a copy into the caller's arena.
                for (int i = 0; i < 150; i++) {
                    assertEquals((byte) (i * 2), output.get(ValueLayout.JAVA_BYTE, i),
                            "Mismatch at index " + i);
                }
            }
        }
    }

    @Test
    void testOverwriteAndDelete {
        try (OffHeapMemoryStore store = new OffHeapMemoryStore();
                Arena arena = Arena.ofConfined()) {

            byte[] key = "mykey".getBytes();

            MemorySegment v1 = arena.allocate(4);
            v1.set(ValueLayout.JAVA_INT, 0, 0xDEAD);
            store.put(key, v1);

            MemorySegment v2 = arena.allocate(4);
            v2.set(ValueLayout.JAVA_INT, 0, 0xBEEF);
            store.put(key, v2);

            Optional<MemorySegment> result = store.getZeroCopy(key, 4, arena);
            assertTrue(result.isPresent());
            assertEquals(0xBEEF, result.get().get(ValueLayout.JAVA_INT, 0));

            // Delete
            store.delete(key);
            Optional<MemorySegment> afterDelete = store.getZeroCopy(key, 4, arena);
            assertTrue(afterDelete.isEmpty());

            // Stats: deletedBytes should be non-zero
            assertTrue(store.deletedBytes() > 0);
        }
    }

    @Test
    void testCompactionReclaimsMemory {
        OffHeapMemoryStore.Config cfg = OffHeapMemoryStore.Config.builder()
                .maxStorageMiB(1L) // 1 MiB — tiny; compaction should trigger quickly
                .build();

        try (OffHeapMemoryStore store = new OffHeapMemoryStore(cfg);
                Arena arena = Arena.ofConfined()) {

            byte[] data = new byte[1024 * 512]; // 512 KiB

            // Write key A and key B — each 512 KiB, total 1 MiB
            MemorySegment segA = arena.allocate(data.length);
            store.put("A".getBytes(), segA);

            MemorySegment segB = arena.allocate(data.length);
            store.put("B".getBytes(), segB);

            // Delete A — half the storage is now dead
            store.delete("A".getBytes());
            assertTrue(store.deletedBytes() > 0, "should track deleted bytes");

            // Force flush/compact
            store.flush();

            // After compaction, deletedBytes should be 0
            assertEquals(0, store.deletedBytes(),
                    "compaction should reset deletedBytes to 0");

            // B should still be readable
            Optional<MemorySegment> bResult = store.getZeroCopy(
                    "B".getBytes(), data.length, arena);
            assertTrue(bResult.isPresent(), "Key B should survive compaction");

            // A should be gone
            Optional<MemorySegment> aResult = store.getZeroCopy(
                    "A".getBytes(), data.length, arena);
            assertTrue(aResult.isEmpty(), "Key A should not survive compaction after delete");

            System.out.println("After compaction: " + store.stats());
        }
    }


    @Test
    void borrowPinsStorageUntilLeaseIsClosed throws Exception {
        try (OffHeapMemoryStore store = new OffHeapMemoryStore();
                Arena inputArena = Arena.ofConfined()) {
            byte[] key = "leased-value".getBytes();
            MemorySegment input = inputArena.allocate(8);
            input.set(ValueLayout.JAVA_LONG, 0, 0x1020304050607080L);
            store.put(key, input);

            var lease = store.borrow(key).orElseThrow();
            assertEquals(8, lease.segment().byteSize());
            assertEquals(0x1020304050607080L,
                    lease.segment().get(ValueLayout.JAVA_LONG, 0));

            CountDownLatch flushStarted = new CountDownLatch(1);
            CountDownLatch flushFinished = new CountDownLatch(1);
            Thread compactor = new Thread(() -> {
                flushStarted.countDown();
                store.flush();
                flushFinished.countDown();
            });
            compactor.start();

            assertTrue(flushStarted.await(1, TimeUnit.SECONDS));
            assertFalse(flushFinished.await(100, TimeUnit.MILLISECONDS),
                    "compaction must wait while a lease is open");
            assertEquals(0x1020304050607080L,
                    lease.segment().get(ValueLayout.JAVA_LONG, 0));

            lease.close();
            assertTrue(flushFinished.await(2, TimeUnit.SECONDS),
                    "compaction should proceed after the lease closes");
            assertTrue(lease.isClosed());
            assertThrows(IllegalStateException.class, lease::segment);
            compactor.join();
        }
    }

    @Test
    void zeroExpectedSizeReturnsWholeLegacyValue {
        try (OffHeapMemoryStore store = new OffHeapMemoryStore();
                Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(32);
            store.put("whole-value".getBytes(), input);

            MemorySegment output = store.getZeroCopy("whole-value".getBytes(), 0, arena).orElseThrow();
            assertEquals(32, output.byteSize());
        }
    }

    @Test
    void testStats {
        try (OffHeapMemoryStore store = new OffHeapMemoryStore();
                Arena arena = Arena.ofConfined()) {

            MemorySegment seg = arena.allocate(1024);
            store.put("stats-key".getBytes(), seg);

            String stats = store.stats();
            assertTrue(stats.contains("entries=1"), stats);
            assertTrue(stats.contains("live="), stats);
            System.out.println("Stats: " + stats);
        }
    }
}
