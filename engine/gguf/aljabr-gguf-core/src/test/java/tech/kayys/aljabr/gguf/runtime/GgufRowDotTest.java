package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufRowDotTest {
    @Test
    void rejectsUnsupportedKnownAndSparseTypeIdsBeforeReadingData() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(1);
            float[] vector = new float[0];

            assertThrows(
                    UnsupportedOperationException.class,
                    () -> GgufRowDot.row(segment, 0, 16, 0, vector, 0));
            assertThrows(
                    UnsupportedOperationException.class,
                    () -> GgufRowDot.row(segment, 0, 4, 0, vector, 0));
        }
    }
}
