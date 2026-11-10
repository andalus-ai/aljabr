package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufKvCacheSequenceLifecycleTest {

    @Test
    void prefillCommitsThePromptLengthOnce() {
        KVCache cache = new KVCache(2, 8, 1, 2);

        GgufKvCacheSequenceLifecycle.completePrefill(cache, 5);

        assertEquals(5, cache.seqLen());
    }

    @Test
    void prefillRejectsNonEmptyCache() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance();

        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheSequenceLifecycle.completePrefill(cache, 3));
        assertEquals(1, cache.seqLen());
    }

    @Test
    void prefillRejectsNegativeTokenCount() {
        KVCache cache = new KVCache(1, 8, 1, 2);

        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheSequenceLifecycle.completePrefill(cache, -1));
        assertEquals(0, cache.seqLen());
    }

    @Test
    void decodeCommitsExactlyOnePositionAfterSuccessfulExecution() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(4);

        GgufKvCacheSequenceLifecycle.completeDecode(cache, 4);

        assertEquals(5, cache.seqLen());
    }

    @Test
    void decodeRejectsUnexpectedPositionWithoutAdvancing() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(4);

        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheSequenceLifecycle.completeDecode(cache, 3));
        assertEquals(4, cache.seqLen());
    }

    @Test
    void decodeRejectsNegativePosition() {
        KVCache cache = new KVCache(1, 8, 1, 2);

        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheSequenceLifecycle.completeDecode(cache, -1));
        assertEquals(0, cache.seqLen());
    }
}
