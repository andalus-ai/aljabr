package tech.kayys.aljabr.gguf.javabackend;

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
}
