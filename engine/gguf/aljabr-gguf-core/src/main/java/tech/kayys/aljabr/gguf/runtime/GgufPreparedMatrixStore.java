package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q2KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q3KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q5KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q6KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q32Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

/**
 * Prepared-matrix cache facade.
 *
 * <p>Each quant family is declared as a {@link GgufSlot}; slot storage and
 * admission stay out of the public {@link GgufTensorOps} surface.</p>
 * @author bhangun
 */
final class GgufPreparedMatrixStore {
    private static final GgufSlot<Q32Matrix, Q32ModelCache> Q32_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q32,
                    Q32ModelCache::new,
                    GgufTensorOps::q32Matrix);
    private static final GgufSlot<Q2KMatrix, Q2KModelCache> Q2K_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q2K,
                    Q2KModelCache::new,
                    GgufTensorOps::q2KMatrix);
    private static final GgufSlot<Q3KMatrix, Q3KModelCache> Q3K_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q3K,
                    Q3KModelCache::new,
                    GgufTensorOps::q3KMatrix);
    private static final GgufSlot<Q4KMatrix, Q4KModelCache> Q4K_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q4K,
                    Q4KModelCache::new,
                    GgufTensorOps::q4KMatrix);
    private static final GgufSlot<Q5KMatrix, Q5KModelCache> Q5K_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q5K,
                    Q5KModelCache::new,
                    GgufTensorOps::q5KMatrix);
    private static final GgufSlot<Q6KMatrix, Q6KModelCache> Q6K_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q6K,
                    Q6KModelCache::new,
                    GgufTensorOps::q6KMatrix);
    private static final GgufSlot<Q8Matrix, Q8ModelCache> Q8_SLOT =
            new GgufSlot<>(
                    GgufPreparedCachePolicy.Family.Q8,
                    Q8ModelCache::new,
                    GgufTensorOps::q8Matrix);
    private static final GgufSlot<?, ?>[] MATRIX_CACHES_BY_BUCKET = matrixCachesByBucket();

    private GgufPreparedMatrixStore() {
    }

    static int clearPreparedMatrixCaches(GGUFModel model) {
        int cleared = 0;
        for (GgufSlot<?, ?> slot : MATRIX_CACHES_BY_BUCKET) {
            cleared += slot.clear(model);
        }
        return cleared;
    }

    static int preparedMatrixCacheSize(GGUFModel model) {
        int cacheSize = 0;
        for (GgufSlot<?, ?> slot : MATRIX_CACHES_BY_BUCKET) {
            cacheSize += slot.size(model);
        }
        return cacheSize;
    }

    static long preparedMatrixCacheBytes(GGUFModel model) {
        long cacheBytes = 0L;
        for (GgufSlot<?, ?> slot : MATRIX_CACHES_BY_BUCKET) {
            cacheBytes += slot.bytes(model);
        }
        return cacheBytes;
    }

    static Q32Matrix q32MatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q32_SLOT.cached(model, tensor);
    }

    static Q2KMatrix q2KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q2K_SLOT.cached(model, tensor);
    }

    static Q3KMatrix q3KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q3K_SLOT.cached(model, tensor);
    }

    static Q4KMatrix q4KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q4K_SLOT.cached(model, tensor);
    }

    static Q5KMatrix q5KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q5K_SLOT.cached(model, tensor);
    }

    static Q6KMatrix q6KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q6K_SLOT.cached(model, tensor);
    }

    static Q8Matrix q8MatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return Q8_SLOT.cached(model, tensor);
    }

    static PreparedMatrix matrixCached(
            GGUFModel model,
            GGUFTensorInfo tensor,
            GgufPreparedCachePolicy.Family family) {
        return cacheSlot(family).cached(model, tensor);
    }

    static PreparedMatrix matrixForMatVec(
            GGUFModel model,
            GGUFTensorInfo tensor,
            GgufKey key,
            GgufPreparedCachePolicy.Family family,
            long maxBytes) {
        return cacheSlot(family).forMatVec(model, tensor, key, maxBytes);
    }

    static int clearMatrixCache(GGUFModel model, GgufPreparedCachePolicy.Family family) {
        return cacheSlot(family).clear(model);
    }

    static int matrixCacheSize(GGUFModel model, GgufPreparedCachePolicy.Family family) {
        return cacheSlot(family).size(model);
    }

    static long matrixCacheBytes(GGUFModel model, GgufPreparedCachePolicy.Family family) {
        return cacheSlot(family).bytes(model);
    }

    private static GgufSlot<?, ?> cacheSlot(GgufPreparedCachePolicy.Family family) {
        return MATRIX_CACHES_BY_BUCKET[family.bucket()];
    }

    private static GgufSlot<?, ?>[] matrixCachesByBucket() {
        GgufSlot<?, ?>[] caches = new GgufSlot<?, ?>[GgufPreparedCachePolicy.preparedMatrixCacheBucketCount()];
        caches[GgufPreparedCachePolicy.Family.Q32.bucket()] = Q32_SLOT;
        caches[GgufPreparedCachePolicy.Family.Q2K.bucket()] = Q2K_SLOT;
        caches[GgufPreparedCachePolicy.Family.Q3K.bucket()] = Q3K_SLOT;
        caches[GgufPreparedCachePolicy.Family.Q4K.bucket()] = Q4K_SLOT;
        caches[GgufPreparedCachePolicy.Family.Q5K.bucket()] = Q5K_SLOT;
        caches[GgufPreparedCachePolicy.Family.Q6K.bucket()] = Q6K_SLOT;
        caches[GgufPreparedCachePolicy.Family.Q8.bucket()] = Q8_SLOT;
        for (GgufSlot<?, ?> cache : caches) {
            if (cache == null) {
                throw new IllegalStateException("Missing GGUF prepared matrix cache bucket");
            }
        }
        return caches;
    }
}
