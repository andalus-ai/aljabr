package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q2KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q3KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q5KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q6KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q32Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Per-model cache for prepared-matrix size estimates.
 *
 * <p>Estimate caching avoids rescanning quantized tensor bytes while planning
 * which matrices are worth preparing for repeated mat-vec calls.</p>
 * @author bhangun
 */
final class PreparedMatrixEstimateModelCache {
    private final Map<GgufKey, Long> estimates = new LinkedHashMap<>();

    Long get(GgufKey key) {
        return estimates.get(key);
    }

    void put(GgufKey key, long estimatedBytes) {
        estimates.put(key, estimatedBytes);
    }

    int size() {
        return estimates.size();
    }
}

/**
 * Insertion-order base cache for prepared GGUF matrices.
 *
 * <p>Generation hits prepared matrices repeatedly, so successful lookups stay
 * read-only instead of reordering the map on every token. The cache still owns
 * eviction and sizing through the shared {@link PreparedMatrix}
 * byte-accounting contract while {@link GgufPreparedCachePolicy} owns admission policy.</p>
 */
class GgufPreparedMatrixCache<K, M extends PreparedMatrix> {
    private final LinkedHashMap<K, M> matrices = new LinkedHashMap<>(16, 0.75f, false);
    private long bytes;

    M get(K key) {
        return matrices.get(key);
    }

    void put(K key, M matrix, long maxBytes) {
        M previous = matrices.put(key, matrix);
        if (previous != null) {
            bytes -= previous.estimatedBytes();
        }
        bytes += matrix.estimatedBytes();
        if (bytes > maxBytes) {
            evictTo(maxBytes);
        }
    }

    int size() {
        return matrices.size();
    }

    long bytes() {
        return bytes;
    }

    void evictTo(long maxBytes) {
        Iterator<Map.Entry<K, M>> iterator = matrices.entrySet().iterator();
        while (bytes > maxBytes && iterator.hasNext()) {
            Map.Entry<K, M> eldest = iterator.next();
            bytes -= eldest.getValue().estimatedBytes();
            iterator.remove();
        }
    }
}

/** Cache for prepared Q4_0/Q4_1/Q5_0/Q5_1 matrices normalized to 32-value blocks. */
final class Q32ModelCache extends GgufPreparedMatrixCache<GgufKey, Q32Matrix> {}

/** Cache for prepared Q2_K matrices. */
final class Q2KModelCache extends GgufPreparedMatrixCache<GgufKey, Q2KMatrix> {}

/** Cache for prepared Q3_K matrices. */
final class Q3KModelCache extends GgufPreparedMatrixCache<GgufKey, Q3KMatrix> {}

/** Cache for prepared Q4_K matrices. */
final class Q4KModelCache extends GgufPreparedMatrixCache<GgufKey, Q4KMatrix> {}

/** Cache for prepared Q5_K matrices. */
final class Q5KModelCache extends GgufPreparedMatrixCache<GgufKey, Q5KMatrix> {}

/** Cache for prepared Q6_K matrices. */
final class Q6KModelCache extends GgufPreparedMatrixCache<GgufKey, Q6KMatrix> {}

/** Cache for prepared Q8_0/Q8_1/Q8_K matrices. */
final class Q8ModelCache extends GgufPreparedMatrixCache<GgufKey, Q8Matrix> {}
