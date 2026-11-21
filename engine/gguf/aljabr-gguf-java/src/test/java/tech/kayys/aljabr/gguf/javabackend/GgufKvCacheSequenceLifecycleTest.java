package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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

    @Test
    void decodeInputValidationAcceptsNextTokenAtAvailablePosition() {
        KVCache cache = new KVCache(2, 8, 1, 4);
        cache.advance(3);

        assertDoesNotThrow(() -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(
                cache, 17, 3, 32, 8));
        assertEquals(3, cache.seqLen());
    }

    @Test
    void decodeInputValidationRejectsNullCacheAndInvalidTokenIds() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(null, 0, 0, 32, 8));

        KVCache cache = new KVCache(1, 8, 1, 4);
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(cache, -1, 0, 32, 8));
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(cache, 32, 0, 32, 8));
        assertEquals(0, cache.seqLen());
    }

    @Test
    void decodeInputValidationRejectsPositionMismatch() {
        KVCache cache = new KVCache(1, 8, 1, 4);
        cache.advance(2);

        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(cache, 1, 1, 32, 8));
        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(cache, 1, 3, 32, 8));
        assertEquals(2, cache.seqLen());
    }

    @Test
    void decodeInputValidationRejectsContextAndCacheCapacityOverflow() {
        KVCache shortCache = new KVCache(1, 2, 1, 4);
        shortCache.advance(2);
        assertThrows(IndexOutOfBoundsException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(shortCache, 1, 2, 32, 8));

        KVCache cache = new KVCache(1, 8, 1, 4);
        cache.advance(4);
        assertThrows(IndexOutOfBoundsException.class,
                () -> GgufKvCacheSequenceLifecycle.validateDecodeInputs(cache, 1, 4, 32, 4));
        assertEquals(4, cache.seqLen());
    }


    @Test
    void sessionDecodeCommitAdvancesSessionAndBackingCacheTogether() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(4);
        KVCacheSession session = new KVCacheSession(cache.asStorage());
        session.commit(4);
        session.write(0, 4, new float[] {1f, 2f}, 0, new float[] {3f, 4f}, 0);

        GgufKvCacheSequenceLifecycle.completeDecode(cache, 4, session);

        assertEquals(5, session.committedLength());
        assertEquals(5, cache.seqLen());
    }

    @Test
    void sessionDecodeCommitRejectsMismatchedPrefixWithoutAdvancingBackingCache() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(4);
        KVCacheSession session = new KVCacheSession(cache.asStorage());
        session.commit(3);

        assertThrows(IllegalStateException.class,
                () -> GgufKvCacheSequenceLifecycle.completeDecode(cache, 4, session));
        assertEquals(4, cache.seqLen());
        assertEquals(3, session.committedLength());
    }

}
