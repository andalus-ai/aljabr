package tech.kayys.aljabr.integration.kv.tensor;

import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Creates independent tensor snapshots from committed KV-cache positions.
 *
 * <p>This adapter deliberately copies data. The current KVCacheStorage
 * contract exposes transfer operations, not stable memory views; claiming
 * zero-copy here would be incorrect for heap storage that may grow or relocate.
 * The returned tensor owns its storage and is independent of the cache session.</p>
 */
public final class KVCacheTensorSnapshots {
    private KVCacheTensorSnapshots() {
    }

    /** Copies committed key vectors into a tensor shaped [tokens, kvHeads, headDimension]. */
    public static Tensor keys(
            KVCacheSession session,
            int layer,
            int firstPosition,
            int tokenCount,
            ComputeBackend backend) {
        return snapshot(session, layer, firstPosition, tokenCount, backend, true);
    }

    /** Copies committed value vectors into a tensor shaped [tokens, kvHeads, headDimension]. */
    public static Tensor values(
            KVCacheSession session,
            int layer,
            int firstPosition,
            int tokenCount,
            ComputeBackend backend) {
        return snapshot(session, layer, firstPosition, tokenCount, backend, false);
    }

    private static Tensor snapshot(
            KVCacheSession session,
            int layer,
            int firstPosition,
            int tokenCount,
            ComputeBackend backend,
            boolean keys) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(backend, "backend");
        if (tokenCount < 0) {
            throw new IllegalArgumentException("tokenCount must not be negative");
        }

        KVCacheLayout layout = session.layout();
        int heads = layout.kvHeads(layer);
        int headDimension = layout.headDimension(layer);
        int elements = Math.multiplyExact(tokenCount, layout.layerStride(layer));
        float[] data = new float[elements];
        if (keys) {
            session.readKeyRange(layer, firstPosition, tokenCount, data, 0);
        } else {
            session.readValueRange(layer, firstPosition, tokenCount, data, 0);
        }
        return TensorFactory.of(data, backend, tokenCount, heads, headDimension);
    }
}
