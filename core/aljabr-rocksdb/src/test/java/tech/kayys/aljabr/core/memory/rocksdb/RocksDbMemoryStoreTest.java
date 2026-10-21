package tech.kayys.aljabr.core.memory.rocksdb;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.core.memory.ManagedArena;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RocksDbMemoryStoreTest {

    @Test
    void testZeroCopyGet(@TempDir Path tempDir) {
        try (RocksDbMemoryStore store = new RocksDbMemoryStore(tempDir.toString())) {
            byte[] key = "test-key".getBytes();

            try (ManagedArena arena = ManagedArena.create()) {
                // Prepare some data
                MemorySegment input = arena.allocate(100);
                for (int i = 0; i < 100; i++) {
                    input.set(ValueLayout.JAVA_BYTE, i, (byte) i);
                }

                // Put
                store.put(key, input);

                // Get Zero Copy
                Optional<MemorySegment> outputOpt = store.getZeroCopy(key, 100, arena.raw());
                assertTrue(outputOpt.isPresent());

                MemorySegment output = outputOpt.get();
                assertEquals(100, output.byteSize());

                // Verify
                for (int i = 0; i < 100; i++) {
                    assertEquals((byte) i, output.get(ValueLayout.JAVA_BYTE, i));
                }
            }
        }
    }
}
