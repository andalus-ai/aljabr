package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufDecodeSessionTest {

    @Test
    void newSessionStartsWithEmptyCommittedPrefix() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        GgufDecodeSession session = new GgufDecodeSession(cache);

        assertEquals(0, session.committedLength());
        assertDoesNotThrow(session::requireEmptyForPrefill);
        assertDoesNotThrow(() -> session.validateDecodePosition(0));
    }

    @Test
    void sessionAttachesToExistingBackingPrefix() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(4);
        GgufDecodeSession session = new GgufDecodeSession(cache);

        assertEquals(4, session.committedLength());
        assertDoesNotThrow(() -> session.validateDecodePosition(4));
        assertThrows(IllegalStateException.class, session::requireEmptyForPrefill);
    }

    @Test
    void sessionRetainsCommittedPrefixAcrossMultipleDecodeSteps() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(3);
        GgufDecodeSession session = new GgufDecodeSession(cache);

        session.cacheSession().write(0, 3, new float[] {1f, 2f}, 0, new float[] {3f, 4f}, 0);
        GgufKvCacheSequenceLifecycle.completeDecode(cache, 3, session.cacheSession());
        assertDoesNotThrow(() -> session.validateDecodePosition(4));

        session.cacheSession().write(0, 4, new float[] {5f, 6f}, 0, new float[] {7f, 8f}, 0);
        GgufKvCacheSequenceLifecycle.completeDecode(cache, 4, session.cacheSession());

        assertEquals(5, session.committedLength());
        assertEquals(5, cache.seqLen());
        assertDoesNotThrow(() -> session.validateDecodePosition(5));
    }

    @Test
    void sessionRejectsPositionThatDiffersFromBackingCache() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(3);
        GgufDecodeSession session = new GgufDecodeSession(cache);

        assertThrows(IllegalStateException.class, () -> session.validateDecodePosition(2));
        assertThrows(IllegalStateException.class, () -> session.validateDecodePosition(4));
    }
}
