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
    void sessionRejectsPositionThatDiffersFromBackingCache() {
        KVCache cache = new KVCache(1, 8, 1, 2);
        cache.advance(3);
        GgufDecodeSession session = new GgufDecodeSession(cache);

        assertThrows(IllegalStateException.class, () -> session.validateDecodePosition(2));
        assertThrows(IllegalStateException.class, () -> session.validateDecodePosition(4));
    }
}
