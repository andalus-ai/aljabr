package tech.kayys.aljabr.gguf.javabackend;

import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

/**
 * Request-scoped GGUF decode state.
 *
 * <p>The session keeps the logical committed prefix alive across prefill and
 * host decode calls. The backing {@link KVCache} remains caller-owned; this
 * wrapper never closes or clears it. A session is single-request state and
 * must not be used concurrently.</p>
 *
 * <p>Native Metal decode continues to use its native cache lifecycle. This
 * object is consumed by the host Java decode path and does not mirror tokens
 * generated exclusively on the native path into host storage.</p>
 */
final class GgufDecodeSession {
    private final KVCache cache;
    private final KVCacheSession cacheSession;

    GgufDecodeSession(KVCache cache) {
        this.cache = Objects.requireNonNull(cache, "cache");
        this.cacheSession = new KVCacheSession(cache.asStorage());
        if (cache.seqLen() > 0) {
            cacheSession.commit(cache.seqLen());
        }
    }

    KVCache cache() {
        return cache;
    }

    KVCacheSession cacheSession() {
        return cacheSession;
    }

    int committedLength() {
        return cacheSession.committedLength();
    }

    void requireEmptyForPrefill() {
        if (cache.seqLen() != 0 || cacheSession.committedLength() != 0) {
            throw new IllegalStateException("prefill requires an empty GGUF decode session");
        }
    }

    void commitPrefill() {
        int sequenceLength = cache.seqLen();
        if (sequenceLength <= 0) {
            throw new IllegalStateException("successful prefill did not advance the backing KV cache");
        }
        if (cacheSession.committedLength() != 0) {
            throw new IllegalStateException("prefill session already has a committed prefix");
        }
        cacheSession.commit(sequenceLength);
    }

    void validateDecodePosition(int position) {
        if (position != cache.seqLen()) {
            throw new IllegalStateException(
                    "decode position " + position + " does not match backing KV cache sequence length "
                            + cache.seqLen());
        }
        if (position != cacheSession.committedLength()) {
            throw new IllegalStateException(
                    "decode position " + position + " does not match request KV cache session length "
                            + cacheSession.committedLength());
        }
    }
}
