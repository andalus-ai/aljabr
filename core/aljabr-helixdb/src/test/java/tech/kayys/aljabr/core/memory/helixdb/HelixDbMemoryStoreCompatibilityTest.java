package tech.kayys.aljabr.core.memory.helixdb;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.memory.BorrowableMemoryStore;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import static org.junit.jupiter.api.Assertions.assertEquals;

class HelixDbMemoryStoreCompatibilityTest {
    @Test
    void legacyNameDelegatesToOffHeapImplementation() throws Exception {
        try (BorrowableMemoryStore store = new HelixDbMemoryStore("ignored");
             Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(4);
            input.set(ValueLayout.JAVA_INT, 0, 42);
            store.put("legacy".getBytes(), input);
            try (var lease = store.borrow("legacy".getBytes()).orElseThrow()) {
                assertEquals(42, lease.segment().get(ValueLayout.JAVA_INT, 0));
            }
        }
    }
}
