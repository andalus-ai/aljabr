package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import java.lang.ref.WeakReference;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * Prepared-matrix byte estimation and estimate-cache bookkeeping.
 * @author bhangun
 */
final class GgufPreparedMatrixEstimator {
    private static final int RECENT_ESTIMATE_SLOTS = 256;
    private static final int RECENT_ESTIMATE_MASK = RECENT_ESTIMATE_SLOTS - 1;
    private static final ThreadLocal<RecentEstimates> RECENT_ESTIMATES =
            ThreadLocal.withInitial(RecentEstimates::new);
    private static final Map<GGUFModel, PreparedMatrixEstimateModelCache> ESTIMATE_CACHE =
            new WeakHashMap<>();
    private static volatile LastEstimate lastEstimate = LastEstimate.empty();
    private static volatile LastEstimate previousEstimate = LastEstimate.empty();
    private static WeakReference<GGUFModel> lastModel = new WeakReference<>(null);
    private static PreparedMatrixEstimateModelCache lastCache;

    private GgufPreparedMatrixEstimator() {
    }

    static long estimate(GGUFTensorInfo tensor) {
        Objects.requireNonNull(tensor, "tensor");
        int typeId = tensor.typeId();
        if (!supportsPreparedMatVecType(typeId)) {
            throw new IllegalArgumentException("Tensor type does not have a prepared matrix cache: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);

        if (supportsQ32PreparedType(typeId)) {
            int totalBlocks = Math.multiplyExact(rows, columns / Q4_0_BLOCK_SIZE);
            return estimateQ32Bytes(totalBlocks, q32PreparedHasBlockBiases(typeId));
        }
        if (typeId == GgmlType.Q2_K.id) {
            return estimateBlockMatrixBytes(rows, columns, QK_K, QK_K, 32);
        }
        if (typeId == GgmlType.Q3_K.id) {
            return estimateBlockMatrixBytes(rows, columns, QK_K, QK_K, 16);
        }
        if (typeId == GgmlType.Q4_K.id || typeId == GgmlType.Q5_K.id) {
            return estimateBlockMatrixBytes(rows, columns, QK_K, QK_K, 16);
        }
        if (typeId == GgmlType.Q6_K.id) {
            return estimateBlockMatrixBytes(rows, columns, QK_K, QK_K, 16);
        }
        if (typeId == GgmlType.NVFP4.id) {
            return estimateBlockMatrixBytes(rows, columns, NVFP4_SUB_BLOCK_SIZE, NVFP4_SUB_BLOCK_SIZE, 1);
        }
        if (typeId == GgmlType.IQ4_XS.id) {
            return estimateBlockMatrixBytes(rows, columns, IQ4_XS_GROUP_SIZE, IQ4_XS_GROUP_SIZE, 1);
        }
        return estimateBlockMatrixBytes(rows, columns, q8BlockSize(typeId), q8BlockSize(typeId), 1);
    }

    static long estimate(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        return estimate(model, tensor, GgufKey.from(tensor));
    }

    static long estimate(GGUFModel model, GGUFTensorInfo tensor, GgufKey key) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        Objects.requireNonNull(key, "key");
        Long cachedEstimate = cached(model, key);
        if (cachedEstimate != null) {
            return cachedEstimate;
        }

        long estimate;
        try {
            estimate = estimateFromData(model, tensor);
        } catch (RuntimeException ignored) {
            estimate = estimate(tensor);
        }

        synchronized (ESTIMATE_CACHE) {
            PreparedMatrixEstimateModelCache modelCache = cacheForCreateLocked(model);
            Long cached = modelCache.get(key);
            if (cached != null) {
                return cached;
            }
            modelCache.put(key, estimate);
            rememberEstimate(model, key, estimate);
            return estimate;
        }
    }

    static int size(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (ESTIMATE_CACHE) {
            PreparedMatrixEstimateModelCache modelCache = cacheForLocked(model);
            return modelCache == null ? 0 : modelCache.size();
        }
    }

    static void remember(
            GGUFModel model,
            GGUFTensorInfo tensor,
            long estimatedBytes) {
        remember(model, GgufKey.from(tensor), estimatedBytes);
    }

    static void rememberKHasMinsHint(
            GGUFModel model,
            GgufKey key,
            int totalBlocks,
            int groupsPerBlock,
            boolean hasGroupMins) {
        remember(model, key, estimateKBytes(totalBlocks, groupsPerBlock, hasGroupMins));
    }

    static void rememberQ32HasBlockBiasHint(
            GGUFModel model,
            GgufKey key,
            int totalBlocks,
            boolean hasBlockBiases) {
        remember(model, key, estimateQ32Bytes(totalBlocks, hasBlockBiases));
    }

    private static void remember(
            GGUFModel model,
            GgufKey key,
            long estimatedBytes) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(key, "key");
        synchronized (ESTIMATE_CACHE) {
            PreparedMatrixEstimateModelCache modelCache = cacheForCreateLocked(model);
            modelCache.put(key, estimatedBytes);
            rememberEstimate(model, key, estimatedBytes);
        }
    }

    static <M extends PreparedMatrix> M remember(
            GGUFModel model,
            GGUFTensorInfo tensor,
            M matrix) {
        Objects.requireNonNull(matrix, "matrix");
        remember(model, tensor, matrix.estimatedBytes());
        return matrix;
    }

    static Boolean cachedKHasMins(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int totalBlocks,
            int groupsPerBlock) {
        return cachedKHasMins(model, GgufKey.from(tensor), totalBlocks, groupsPerBlock);
    }

    static Boolean cachedKHasMins(
            GGUFModel model,
            GgufKey key,
            int totalBlocks,
            int groupsPerBlock) {
        Long estimate = cached(model, key);
        if (estimate == null) {
            return null;
        }
        long withoutMins = estimateKBytes(totalBlocks, groupsPerBlock, false);
        if (estimate == withoutMins) {
            return Boolean.FALSE;
        }
        long withMins = estimateKBytes(totalBlocks, groupsPerBlock, true);
        return estimate == withMins ? Boolean.TRUE : null;
    }

    static Boolean cachedQ32HasBlockBiases(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int totalBlocks) {
        return cachedQ32HasBlockBiases(model, GgufKey.from(tensor), totalBlocks);
    }

    static Boolean cachedQ32HasBlockBiases(
            GGUFModel model,
            GgufKey key,
            int totalBlocks) {
        Long estimate = cached(model, key);
        if (estimate == null) {
            return null;
        }
        long withoutBiases = estimateQ32Bytes(totalBlocks, false);
        if (estimate == withoutBiases) {
            return Boolean.FALSE;
        }
        long withBiases = estimateQ32Bytes(totalBlocks, true);
        return estimate == withBiases ? Boolean.TRUE : null;
    }

    private static Long cached(GGUFModel model, GgufKey key) {
        Long recent = recentEstimate(model, key);
        if (recent != null) {
            return recent;
        }
        synchronized (ESTIMATE_CACHE) {
            PreparedMatrixEstimateModelCache modelCache = cacheForLocked(model);
            Long estimate = modelCache == null ? null : modelCache.get(key);
            if (estimate != null) {
                rememberEstimate(model, key, estimate);
            }
            return estimate;
        }
    }

    private static Long recentEstimate(GGUFModel model, GgufKey key) {
        RecentEstimates recent = RECENT_ESTIMATES.get();
        Long estimate = recent.get(model, key);
        if (estimate != null) {
            return estimate;
        }
        LastEstimate last = lastEstimate;
        estimate = estimateFrom(last, model, key);
        if (estimate != null) {
            rememberRecentEstimate(recent, model, key, estimate);
            return estimate;
        }
        if (last.isStale()) {
            lastEstimate = LastEstimate.empty();
        }
        LastEstimate previous = previousEstimate;
        estimate = estimateFrom(previous, model, key);
        if (estimate != null) {
            rememberRecentEstimate(recent, model, key, estimate);
            return estimate;
        }
        if (previous.isStale()) {
            previousEstimate = LastEstimate.empty();
        }
        return null;
    }

    private static Long estimateFrom(LastEstimate estimate, GGUFModel model, GgufKey key) {
        GGUFModel cachedModel = estimate.model().get();
        return cachedModel == model && estimate.key() == key ? estimate.estimatedBytes() : null;
    }

    private static void rememberEstimate(GGUFModel model, GgufKey key, long estimatedBytes) {
        rememberRecentEstimate(model, key, estimatedBytes);
        LastEstimate last = lastEstimate;
        GGUFModel cachedModel = last.model().get();
        if (cachedModel == model && last.key() == key) {
            return;
        }
        if (cachedModel != null && last.key() != null) {
            previousEstimate = last;
        }
        lastEstimate = new LastEstimate(new WeakReference<>(model), key, estimatedBytes);
    }

    static int recentEstimateCacheSize() {
        return RECENT_ESTIMATES.get().size();
    }

    static int recentEstimateFastCacheSize() {
        return RECENT_ESTIMATES.get().fastSize();
    }

    static void clearRecentEstimateCache() {
        RECENT_ESTIMATES.get().clear();
    }

    private static void rememberRecentEstimate(GGUFModel model, GgufKey key, long estimatedBytes) {
        rememberRecentEstimate(RECENT_ESTIMATES.get(), model, key, estimatedBytes);
    }

    private static void rememberRecentEstimate(
            RecentEstimates recent,
            GGUFModel model,
            GgufKey key,
            long estimatedBytes) {
        recent.put(model, key, estimatedBytes);
    }

    private static PreparedMatrixEstimateModelCache cacheForLocked(GGUFModel model) {
        GGUFModel cachedModel = lastModel.get();
        if (cachedModel == model) {
            return lastCache;
        }
        if (cachedModel == null && lastCache != null) {
            lastCache = null;
        }
        PreparedMatrixEstimateModelCache modelCache = ESTIMATE_CACHE.get(model);
        if (modelCache != null) {
            rememberLocked(model, modelCache);
        }
        return modelCache;
    }

    private static PreparedMatrixEstimateModelCache cacheForCreateLocked(GGUFModel model) {
        PreparedMatrixEstimateModelCache modelCache = cacheForLocked(model);
        if (modelCache != null) {
            return modelCache;
        }
        PreparedMatrixEstimateModelCache created = new PreparedMatrixEstimateModelCache();
        ESTIMATE_CACHE.put(model, created);
        rememberLocked(model, created);
        return created;
    }

    private static void rememberLocked(GGUFModel model, PreparedMatrixEstimateModelCache modelCache) {
        lastModel = new WeakReference<>(model);
        lastCache = modelCache;
    }

    private record LastEstimate(
            WeakReference<GGUFModel> model,
            GgufKey key,
            long estimatedBytes) {
        static LastEstimate empty() {
            return new LastEstimate(new WeakReference<>(null), null, 0L);
        }

        private boolean isStale() {
            return key != null && model.get() == null;
        }
    }

    private static final class RecentEstimates {
        private final RecentEstimate[] estimates = new RecentEstimate[RECENT_ESTIMATE_SLOTS];
        private RecentEstimate last;

        private Long get(GGUFModel model, GgufKey key) {
            Long estimatedBytes = estimatedBytesIfMatches(last, model, key);
            if (estimatedBytes != null) {
                return estimatedBytes;
            }
            int index = index(model, key);
            RecentEstimate estimate = estimates[index];
            estimatedBytes = estimatedBytesIfMatches(estimate, model, key);
            if (estimatedBytes != null) {
                last = estimate;
                return estimatedBytes;
            }
            if (estimate != null && recentExpired(estimate)) {
                estimates[index] = null;
            }
            return null;
        }

        private void put(GGUFModel model, GgufKey key, long estimatedBytes) {
            RecentEstimate estimate = new RecentEstimate(new WeakReference<>(model), key, estimatedBytes);
            last = estimate;
            estimates[index(model, key)] = estimate;
        }

        private int size() {
            int size = 0;
            for (int index = 0; index < estimates.length; index++) {
                RecentEstimate estimate = estimates[index];
                if (estimate == null) {
                    continue;
                }
                if (estimate.model().get() == null) {
                    estimates[index] = null;
                } else {
                    size++;
                }
            }
            return size;
        }

        private int fastSize() {
            if (last == null) {
                return 0;
            }
            if (recentExpired(last)) {
                last = null;
                return 0;
            }
            return 1;
        }

        private void clear() {
            last = null;
            for (int index = 0; index < estimates.length; index++) {
                estimates[index] = null;
            }
        }

        private static Long estimatedBytesIfMatches(RecentEstimate estimate, GGUFModel model, GgufKey key) {
            if (estimate == null) {
                return null;
            }
            GGUFModel cachedModel = estimate.model().get();
            return cachedModel == model && estimate.key() == key ? estimate.estimatedBytes() : null;
        }

        private static boolean recentExpired(RecentEstimate estimate) {
            return estimate.model().get() == null;
        }

        private static int index(GGUFModel model, GgufKey key) {
            int hash = System.identityHashCode(model);
            hash = 31 * hash + key.hashCode();
            return hash & RECENT_ESTIMATE_MASK;
        }
    }

    private record RecentEstimate(
            WeakReference<GGUFModel> model,
            GgufKey key,
            long estimatedBytes) {
    }

    private static long estimateFromData(GGUFModel model, GGUFTensorInfo tensor) {
        int typeId = tensor.typeId();
        if (typeId != GgmlType.Q4_1.id
                && typeId != GgmlType.Q5_1.id
                && typeId != GgmlType.Q2_K.id
                && typeId != GgmlType.Q4_K.id
                && typeId != GgmlType.Q5_K.id) {
            return estimate(tensor);
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        MemorySegment source = GgufTensorData.tensorData(model, tensor);
        if (typeId == GgmlType.Q4_1.id) {
            int totalBlocks = Math.multiplyExact(rows, columns / Q4_0_BLOCK_SIZE);
            return estimateQ32Bytes(totalBlocks, GgufQ32Meta.q4_1BlocksHaveBiases(source, totalBlocks));
        }
        if (typeId == GgmlType.Q5_1.id) {
            int totalBlocks = Math.multiplyExact(rows, columns / Q4_0_BLOCK_SIZE);
            return estimateQ32Bytes(totalBlocks, GgufQ32Meta.q5_1BlocksHaveBiases(source, totalBlocks));
        }
        int totalBlocks = Math.multiplyExact(rows, columns / QK_K);
        if (typeId == GgmlType.Q2_K.id) {
            return estimateKBytes(totalBlocks, 16, GgufKMeta.q2BlocksHaveMins(source, totalBlocks));
        }
        if (typeId == GgmlType.Q4_K.id) {
            return estimateKBytes(totalBlocks, 8, GgufKMeta.q4BlocksHaveMins(source, totalBlocks));
        }
        return estimateKBytes(totalBlocks, 8, GgufKMeta.q5BlocksHaveMins(source, totalBlocks));
    }

    private static long estimateQ32Bytes(int totalBlocks, boolean hasBlockBiases) {
        long quantBytes = Math.multiplyExact((long) totalBlocks, (long) Q4_0_BLOCK_SIZE);
        long scaleBytes = Math.multiplyExact((long) totalBlocks, (long) Float.BYTES);
        long bytes = Math.addExact(quantBytes, scaleBytes);
        return hasBlockBiases ? Math.addExact(bytes, scaleBytes) : bytes;
    }

    private static long estimateKBytes(int totalBlocks, int groupsPerBlock, boolean hasGroupMins) {
        long quantBytes = Math.multiplyExact((long) totalBlocks, QK_K);
        long scaleBytes = Math.multiplyExact(
                Math.multiplyExact((long) totalBlocks, groupsPerBlock),
                (long) Float.BYTES);
        long bytes = Math.addExact(quantBytes, scaleBytes);
        return hasGroupMins ? Math.addExact(bytes, scaleBytes) : bytes;
    }

    private static long estimateBlockMatrixBytes(
            int rows,
            int columns,
            int groupSize,
            int quantBytesPerGroup,
            int floatScalesPerGroup) {
        int groupsPerRow = columns / groupSize;
        long groups = Math.multiplyExact((long) rows, groupsPerRow);
        long bytesPerGroup = Math.addExact(
                quantBytesPerGroup,
                Math.multiplyExact((long) floatScalesPerGroup, Float.BYTES));
        return Math.multiplyExact(groups, bytesPerGroup);
    }
}
