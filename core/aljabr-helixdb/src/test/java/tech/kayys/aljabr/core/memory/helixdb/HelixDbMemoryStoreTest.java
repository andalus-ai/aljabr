package tech.kayys.aljabr.core.memory.helixdb;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.core.memory.ManagedArena;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class HelixDbMemoryStoreTest {

    @Test
    void testZeroCopyGet(@TempDir Path tempDir) {
        try (HelixDbMemoryStore store = new HelixDbMemoryStore(tempDir.toString())) {
            byte[] key = "test-key-helix".getBytes();

            try (ManagedArena arena = ManagedArena.create()) {
                // Prepare some data
                MemorySegment input = arena.allocate(150);
                for (int i = 0; i < 150; i++) {
                    input.set(ValueLayout.JAVA_BYTE, i, (byte) (i * 2));
                }

                // Put
                store.put(key, input);

                // Get Zero Copy
                Optional<MemorySegment> outputOpt = store.getZeroCopy(key, 150, arena.raw());
                assertTrue(outputOpt.isPresent());

                MemorySegment output = outputOpt.get();
                assertEquals(150, output.byteSize());

                // Verify byte values — this should be a pointer into HelixDB's storage,
                // not a copy into the caller's arena.
                for (int i = 0; i < 150; i++) {
                    assertEquals((byte) (i * 2), output.get(ValueLayout.JAVA_BYTE, i),
                            "Mismatch at index " + i);
                }
            }
        }
    }

    @Test
    void testOverwriteAndDelete(@TempDir Path tempDir) {
        try (HelixDbMemoryStore store = new HelixDbMemoryStore(tempDir.toString());
                ManagedArena arena = ManagedArena.create()) {

            byte[] key = "mykey".getBytes();

            MemorySegment v1 = arena.allocate(4);
            v1.set(ValueLayout.JAVA_INT, 0, 0xDEAD);
            store.put(key, v1);

            MemorySegment v2 = arena.allocate(4);
            v2.set(ValueLayout.JAVA_INT, 0, 0xBEEF);
            store.put(key, v2);

            Optional<MemorySegment> result = store.getZeroCopy(key, 4, arena.raw());
            assertTrue(result.isPresent());
            assertEquals(0xBEEF, result.get().get(ValueLayout.JAVA_INT, 0));

            // Delete
            store.delete(key);
            Optional<MemorySegment> afterDelete = store.getZeroCopy(key, 4, arena.raw());
            assertTrue(afterDelete.isEmpty());

            // Stats: deletedBytes should be non-zero
            assertTrue(store.deletedBytes() > 0);
        }
    }

    @Test
    void testCompactionReclaimsMemory(@TempDir Path tempDir) {
        HelixDbMemoryStore.Config cfg = HelixDbMemoryStore.Config.builder()
                .maxStorageMiB(1L) // 1 MiB — tiny; compaction should trigger quickly
                .build();

        try (HelixDbMemoryStore store = new HelixDbMemoryStore(tempDir.toString(), cfg);
                ManagedArena arena = ManagedArena.create()) {

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
                    "B".getBytes(), data.length, arena.raw());
            assertTrue(bResult.isPresent(), "Key B should survive compaction");

            // A should be gone
            Optional<MemorySegment> aResult = store.getZeroCopy(
                    "A".getBytes(), data.length, arena.raw());
            assertTrue(aResult.isEmpty(), "Key A should not survive compaction after delete");

            System.out.println("After compaction: " + store.stats());
        }
    }

    @Test
    void testStats(@TempDir Path tempDir) {
        try (HelixDbMemoryStore store = new HelixDbMemoryStore(tempDir.toString());
                ManagedArena arena = ManagedArena.create()) {

            MemorySegment seg = arena.allocate(1024);
            store.put("stats-key".getBytes(), seg);

            String stats = store.stats();
            assertTrue(stats.contains("entries=1"), stats);
            assertTrue(stats.contains("live="), stats);
            System.out.println("Stats: " + stats);
        }
    }
}
