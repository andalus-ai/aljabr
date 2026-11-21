package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

/**
 * Owns the sequence-position commit boundary for GGUF prefill and decode.
 *
 * <p>Layer writes never advance the cache. A prefill commits its complete
 * prompt length once, and a successful decode commits exactly one position.
 * This applies equally to host execution and native accelerator execution.</p>
 */
final class GgufKvCacheSequenceLifecycle {
    private GgufKvCacheSequenceLifecycle() {
    }

    static void validateDecodeInputs(KVCache cache, int tokenId, int position, int vocabSize, int contextLength) {
        if (cache == null) {
            throw new IllegalArgumentException("KV cache must not be null");
        }
        if (vocabSize <= 0 || tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("tokenId is outside the model vocabulary: " + tokenId);
        }
        if (contextLength <= 0 || position < 0 || position >= contextLength) {
            throw new IndexOutOfBoundsException("decode position is outside the model context: " + position);
        }
        if (position != cache.seqLen()) {
            throw new IllegalStateException(
                    "decode position " + position + " does not match KV cache sequence length " + cache.seqLen());
        }
        if (position >= cache.capacity()) {
            throw new IndexOutOfBoundsException("decode position exceeds KV cache capacity: " + position);
        }
    }

    static void completePrefill(KVCache cache, int tokenCount) {
        if (tokenCount < 0) {
            throw new IllegalArgumentException("tokenCount must be non-negative");
        }
        if (cache.seqLen() != 0) {
            throw new IllegalStateException(
                    "prefill requires an empty KV cache; current sequence length is " + cache.seqLen());
        }
        cache.advance(tokenCount);
    }

    static void completeDecode(KVCache cache, int position) {
        if (position < 0) {
            throw new IllegalArgumentException("position must be non-negative");
        }
        if (cache.seqLen() != position) {
            throw new IllegalStateException(
                    "decode position " + position + " does not match KV cache sequence length " + cache.seqLen());
        }
        cache.advance();
    }

    /**
     * Commits a host decode step to both its logical session and the legacy
     * GGUF cache sequence pointer. The session must describe the same prefix
     * as the backing cache before the current token is published.
     */
    static void completeDecode(KVCache cache, int position, KVCacheSession session) {
        if (session == null) {
            throw new IllegalArgumentException("KV cache session must not be null");
        }
        if (session.committedLength() != position) {
            throw new IllegalStateException(
                    "decode position " + position + " does not match KV cache session length "
                            + session.committedLength());
        }
        if (cache == null || cache.seqLen() != position) {
            throw new IllegalStateException(
                    "decode position " + position + " does not match backing KV cache sequence length "
                            + (cache == null ? "null" : cache.seqLen()));
        }
        session.commit(position + 1);
        cache.advance();
    }
}
