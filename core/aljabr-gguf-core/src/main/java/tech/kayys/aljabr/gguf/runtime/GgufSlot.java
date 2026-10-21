package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.function.Supplier;

/**
 * One prepared-matrix cache slot for a GGUF quant family.
 *
 * <p>The slot owns weak per-model cache storage plus the prepare/admit path for
 * a single {@link GgufPreparedCachePolicy.Family}; {@link GgufPreparedMatrixStore} keeps only the public
 * routing facade.</p>
 * @author bhangun
 */
final class GgufSlot<M extends PreparedMatrix, C extends GgufPreparedMatrixCache<GgufKey, M>> {
    private static final int RECENT_MATRIX_SLOTS = 256;
    private static final int RECENT_MATRIX_MASK = RECENT_MATRIX_SLOTS - 1;
    private static final int RECENT_REJECT_SLOTS = 256;
    private static final int RECENT_REJECT_MASK = RECENT_REJECT_SLOTS - 1;
    private static final PreparedMatrix REJECTED = () -> -1L;
    private static final ThreadLocal<RecentMatrices> RECENT_MATRICES =
            ThreadLocal.withInitial(RecentMatrices::new);
    private static final ThreadLocal<RecentRejects> RECENT_REJECTS =
            ThreadLocal.withInitial(RecentRejects::new);

    private final GgufPreparedCachePolicy.Family family;
    private final Map<GGUFModel, C> models = new WeakHashMap<>();
    private final Supplier<C> cacheFactory;
    private final Preparer<M> preparer;
    private WeakReference<GGUFModel> lastModel = new WeakReference<>(null);
    private C lastCache;
    private volatile long cacheVersion;
    private volatile long matrixCacheVersion;
    private volatile LastMatrixHit<M> lastHit = LastMatrixHit.empty();
    private volatile LastMatrixHit<M> previousHit = LastMatrixHit.empty();
    private volatile LastMatrixReject lastReject = LastMatrixReject.empty();
    private volatile LastMatrixReject previousReject = LastMatrixReject.empty();

    GgufSlot(
            GgufPreparedCachePolicy.Family family,
            Supplier<C> cacheFactory,
            Preparer<M> preparer) {
        this.family = Objects.requireNonNull(family, "family");
        this.cacheFactory = Objects.requireNonNull(cacheFactory, "cacheFactory");
        this.preparer = Objects.requireNonNull(preparer, "preparer");
    }

    static int recentMatrixFastCacheSize() {
        return RECENT_MATRICES.get().fastSize();
    }

    static int recentRejectFastCacheSize() {
        return RECENT_REJECTS.get().fastSize();
    }

    static void clearRecentSlotCaches() {
        RECENT_MATRICES.get().clear();
        RECENT_REJECTS.get().clear();
    }

    int clear(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (models) {
            C removed = models.remove(model);
            forgetLocked(model, removed);
            invalidateFastStateLocked(model);
            return removed == null ? 0 : removed.size();
        }
    }

    int size(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (models) {
            C cache = cacheForLocked(model);
            return cache == null ? 0 : cache.size();
        }
    }

    long bytes(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        synchronized (models) {
            C cache = cacheForLocked(model);
            return cache == null ? 0L : cache.bytes();
        }
    }

    M cached(GGUFModel model, GGUFTensorInfo tensor) {
        return cached(model, GgufKey.from(tensor), family.maxBytes(), () -> preparer.prepare(model, tensor));
    }

    M forMatVec(GGUFModel model, GGUFTensorInfo tensor, long maxBytes) {
        return forMatVec(model, tensor, GgufKey.from(tensor), maxBytes);
    }

    M forMatVec(GGUFModel model, GGUFTensorInfo tensor, GgufKey key, long maxBytes) {
        if (maxBytes <= 0) {
            return null;
        }
        if (recentReject(model, key, maxBytes)) {
            return null;
        }
        M cached = cachedAfterEvict(model, key, maxBytes);
        if (cached != null) {
            return cached;
        }

        long estimatedBytes = GgufPreparationPlan.preparedMatrixAdmitBytes(model, tensor, key, maxBytes);
        if (estimatedBytes < 0) {
            rememberReject(model, key, maxBytes);
            return null;
        }
        PreparedMatrix admission = admitAfterEvict(model, key, maxBytes, estimatedBytes);
        if (admission == REJECTED) {
            rememberReject(model, key, maxBytes);
            return null;
        }
        if (admission != null) {
            return admitted(admission);
        }
        return prepareAndCache(model, key, maxBytes, () -> preparer.prepare(model, tensor));
    }

    private M cached(GGUFModel model, GgufKey key, long maxBytes, Supplier<M> prepare) {
        Objects.requireNonNull(model, "model");
        if (maxBytes <= 0) {
            synchronized (models) {
                models.remove(model);
                forgetLocked(model, null);
                invalidateFastStateLocked(model);
            }
            return prepare.get();
        }
        M cached = cachedAfterEvict(model, key, maxBytes);
        if (cached != null) {
            return cached;
        }

        return prepareAndCache(model, key, maxBytes, prepare);
    }

    private M cachedAfterEvict(GGUFModel model, GgufKey key, long maxBytes) {
        M hit = recentHit(model, key, maxBytes);
        if (hit != null) {
            return hit;
        }
        synchronized (models) {
            C cache = cacheForLocked(model);
            if (cache == null) {
                return null;
            }
            if (cache.bytes() > maxBytes) {
                cache.evictTo(maxBytes);
                invalidateFastStateLocked(model);
            }
            M cached = cache.get(key);
            if (cached != null) {
                rememberHit(model, key, maxBytes, cached);
            }
            return cached;
        }
    }

    private PreparedMatrix admitAfterEvict(
            GGUFModel model,
            GgufKey key,
            long maxBytes,
            long estimatedBytes) {
        synchronized (models) {
            C cache = cacheForLocked(model);
            if (cache == null) {
                return null;
            }
            if (cache.bytes() > maxBytes) {
                cache.evictTo(maxBytes);
                invalidateFastStateLocked(model);
            }
            M cached = cache.get(key);
            if (cached != null) {
                rememberHit(model, key, maxBytes, cached);
                return cached;
            }
            return cache.bytes() <= maxBytes - estimatedBytes ? null : REJECTED;
        }
    }

    private M putIfAbsent(GGUFModel model, GgufKey key, long maxBytes, M prepared) {
        synchronized (models) {
            C cache = models.computeIfAbsent(model, ignored -> cacheFactory.get());
            rememberLocked(model, cache);
            M cached = cache.get(key);
            if (cached != null) {
                rememberHit(model, key, maxBytes, cached);
                return cached;
            }
            boolean mayEvict = cache.bytes() > maxBytes - prepared.estimatedBytes();
            if (mayEvict) {
                invalidateFastStateLocked(model);
            }
            cache.put(key, prepared, maxBytes);
            if (!mayEvict) {
                invalidateRejectsLocked(model);
            }
            rememberHit(model, key, maxBytes, prepared);
            return prepared;
        }
    }

    private M prepareAndCache(GGUFModel model, GgufKey key, long maxBytes, Supplier<M> prepare) {
        M prepared = prepare.get();
        if (prepared.estimatedBytes() > maxBytes) {
            rememberReject(model, key, maxBytes);
            return prepared;
        }
        return putIfAbsent(model, key, maxBytes, prepared);
    }

    private C cacheForLocked(GGUFModel model) {
        GGUFModel cachedModel = lastModel.get();
        if (cachedModel == model) {
            return lastCache;
        }
        if (cachedModel == null && lastCache != null) {
            lastCache = null;
        }
        C cache = models.get(model);
        if (cache != null) {
            rememberLocked(model, cache);
        }
        return cache;
    }

    private void rememberLocked(GGUFModel model, C cache) {
        lastModel = new WeakReference<>(model);
        lastCache = cache;
    }

    private void forgetLocked(GGUFModel model, C removed) {
        GGUFModel cachedModel = lastModel.get();
        if (cachedModel == model || cachedModel == null || (removed != null && removed == lastCache)) {
            lastModel = new WeakReference<>(null);
            lastCache = null;
        }
    }

    private M recentHit(GGUFModel model, GgufKey key, long maxBytes) {
        M matrix = recentMatrix(model, key, maxBytes);
        if (matrix != null) {
            return matrix;
        }
        LastMatrixHit<M> last = lastHit;
        matrix = matrixFrom(last, model, key, maxBytes);
        if (matrix != null) {
            rememberRecentMatrix(model, key, maxBytes, matrix);
            return matrix;
        }
        if (last.isStale()) {
            lastHit = LastMatrixHit.empty();
        }
        LastMatrixHit<M> previous = previousHit;
        matrix = matrixFrom(previous, model, key, maxBytes);
        if (matrix != null) {
            rememberRecentMatrix(model, key, maxBytes, matrix);
            return matrix;
        }
        if (previous.isStale()) {
            previousHit = LastMatrixHit.empty();
        }
        return null;
    }

    private static <M extends PreparedMatrix> M matrixFrom(
            LastMatrixHit<M> hit,
            GGUFModel model,
            GgufKey key,
            long maxBytes) {
        GGUFModel cachedModel = hit.model().get();
        return cachedModel == model && hit.key() == key && hit.maxBytes() == maxBytes ? hit.matrix() : null;
    }

    private void rememberHit(GGUFModel model, GgufKey key, long maxBytes, M matrix) {
        rememberRecentMatrix(model, key, maxBytes, matrix);
        LastMatrixHit<M> last = lastHit;
        GGUFModel cachedModel = last.model().get();
        if (cachedModel == model && last.key() == key && last.maxBytes() == maxBytes) {
            return;
        }
        if (cachedModel != null && last.matrix() != null) {
            previousHit = last;
        }
        lastHit = new LastMatrixHit<>(new WeakReference<>(model), key, maxBytes, matrix);
    }

    @SuppressWarnings("unchecked")
    private M recentMatrix(GGUFModel model, GgufKey key, long maxBytes) {
        return (M) RECENT_MATRICES.get().get(this, model, key, maxBytes, matrixCacheVersion);
    }

    private void rememberRecentMatrix(GGUFModel model, GgufKey key, long maxBytes, M matrix) {
        RECENT_MATRICES.get().put(this, model, key, maxBytes, matrixCacheVersion, matrix);
    }

    private boolean recentReject(GGUFModel model, GgufKey key, long maxBytes) {
        long version = cacheVersion;
        if (RECENT_REJECTS.get().contains(this, model, key, maxBytes, version)) {
            return true;
        }
        LastMatrixReject last = lastReject;
        if (last.matches(model, key, maxBytes, version)) {
            rememberRecentReject(model, key, maxBytes, version);
            return true;
        }
        if (last.isStale()) {
            lastReject = LastMatrixReject.empty();
        }
        LastMatrixReject previous = previousReject;
        if (previous.matches(model, key, maxBytes, version)) {
            rememberRecentReject(model, key, maxBytes, version);
            return true;
        }
        if (previous.isStale()) {
            previousReject = LastMatrixReject.empty();
        }
        return false;
    }

    private void rememberReject(GGUFModel model, GgufKey key, long maxBytes) {
        long version = cacheVersion;
        rememberRecentReject(model, key, maxBytes, version);
        LastMatrixReject last = lastReject;
        if (last.matches(model, key, maxBytes, version)) {
            return;
        }
        if (!last.isEmpty()) {
            previousReject = last;
        }
        lastReject = new LastMatrixReject(new WeakReference<>(model), key, maxBytes, version);
    }

    private void rememberRecentReject(GGUFModel model, GgufKey key, long maxBytes, long version) {
        RECENT_REJECTS.get().put(this, model, key, maxBytes, version);
    }

    private void invalidateFastStateLocked(GGUFModel model) {
        cacheVersion++;
        matrixCacheVersion++;
        if (hitBelongsTo(lastHit, model)) {
            lastHit = LastMatrixHit.empty();
        }
        if (hitBelongsTo(previousHit, model)) {
            previousHit = LastMatrixHit.empty();
        }
        if (rejectBelongsTo(lastReject, model)) {
            lastReject = LastMatrixReject.empty();
        }
        if (rejectBelongsTo(previousReject, model)) {
            previousReject = LastMatrixReject.empty();
        }
    }

    private void invalidateRejectsLocked(GGUFModel model) {
        cacheVersion++;
        if (rejectBelongsTo(lastReject, model)) {
            lastReject = LastMatrixReject.empty();
        }
        if (rejectBelongsTo(previousReject, model)) {
            previousReject = LastMatrixReject.empty();
        }
    }

    private static boolean hitBelongsTo(LastMatrixHit<?> hit, GGUFModel model) {
        GGUFModel cachedModel = hit.model().get();
        return cachedModel == model || (cachedModel == null && hit.matrix() != null);
    }

    private static boolean rejectBelongsTo(LastMatrixReject reject, GGUFModel model) {
        GGUFModel cachedModel = reject.model().get();
        return cachedModel == model || (cachedModel == null && reject.key() != null);
    }

    @FunctionalInterface
    interface Preparer<M extends PreparedMatrix> {
        M prepare(GGUFModel model, GGUFTensorInfo tensor);
    }

    private record LastMatrixHit<M extends PreparedMatrix>(
            WeakReference<GGUFModel> model,
            GgufKey key,
            long maxBytes,
            M matrix) {
        static <M extends PreparedMatrix> LastMatrixHit<M> empty() {
            return new LastMatrixHit<>(new WeakReference<>(null), null, 0L, null);
        }

        private boolean isStale() {
            return matrix != null && model.get() == null;
        }
    }

    private record LastMatrixReject(
            WeakReference<GGUFModel> model,
            GgufKey key,
            long maxBytes,
            long version) {
        static LastMatrixReject empty() {
            return new LastMatrixReject(new WeakReference<>(null), null, 0L, -1L);
        }

        private boolean matches(GGUFModel model, GgufKey key, long maxBytes, long version) {
            return this.model.get() == model
                    && this.key == key
                    && this.maxBytes == maxBytes
                    && this.version == version;
        }

        private boolean isEmpty() {
            return key == null;
        }

        private boolean isStale() {
            return key != null && model.get() == null;
        }
    }

    @SuppressWarnings("unchecked")
    private M admitted(PreparedMatrix matrix) {
        return (M) matrix;
    }

    private static final class RecentMatrices {
        private final RecentMatrix[] matrices = new RecentMatrix[RECENT_MATRIX_SLOTS];
        private RecentMatrix last;

        private PreparedMatrix get(
                GgufSlot<?, ?> slot,
                GGUFModel model,
                GgufKey key,
                long maxBytes,
                long version) {
            PreparedMatrix matrix = matrixIfMatches(last, slot, model, key, maxBytes, version);
            if (matrix != null) {
                return matrix;
            }
            if (last != null && matrixExpired(last)) {
                last = null;
            }
            int index = index(slot, model, key, maxBytes);
            RecentMatrix recent = matrices[index];
            matrix = matrixIfMatches(recent, slot, model, key, maxBytes, version);
            if (matrix != null) {
                last = recent;
                return matrix;
            }
            if (recent != null && (matrixExpired(recent) || recent.version() != version)) {
                matrices[index] = null;
            }
            return null;
        }

        private void put(
                GgufSlot<?, ?> slot,
                GGUFModel model,
                GgufKey key,
                long maxBytes,
                long version,
                PreparedMatrix matrix) {
            RecentMatrix recent =
                    new RecentMatrix(slot, new WeakReference<>(model), key, maxBytes, version, new WeakReference<>(matrix));
            last = recent;
            matrices[index(slot, model, key, maxBytes)] = recent;
        }

        private int fastSize() {
            if (last == null) {
                return 0;
            }
            if (matrixExpired(last)) {
                last = null;
                return 0;
            }
            return 1;
        }

        private void clear() {
            last = null;
            for (int index = 0; index < matrices.length; index++) {
                matrices[index] = null;
            }
        }

        private static PreparedMatrix matrixIfMatches(
                RecentMatrix recent,
                GgufSlot<?, ?> slot,
                GGUFModel model,
                GgufKey key,
                long maxBytes,
                long version) {
            if (recent == null) {
                return null;
            }
            GGUFModel cachedModel = recent.model().get();
            PreparedMatrix matrix = recent.matrix().get();
            if (cachedModel == model
                    && recent.slot() == slot
                    && recent.key() == key
                    && recent.maxBytes() == maxBytes
                    && recent.version() == version
                    && matrix != null) {
                return matrix;
            }
            return null;
        }

        private static boolean matrixExpired(RecentMatrix recent) {
            return recent.model().get() == null || recent.matrix().get() == null;
        }

        private static int index(GgufSlot<?, ?> slot, GGUFModel model, GgufKey key, long maxBytes) {
            int hash = System.identityHashCode(slot);
            hash = 31 * hash + System.identityHashCode(model);
            hash = 31 * hash + key.hashCode();
            hash = 31 * hash + Long.hashCode(maxBytes);
            return hash & RECENT_MATRIX_MASK;
        }
    }

    private record RecentMatrix(
            GgufSlot<?, ?> slot,
            WeakReference<GGUFModel> model,
            GgufKey key,
            long maxBytes,
            long version,
            WeakReference<PreparedMatrix> matrix) {
    }

    private static final class RecentRejects {
        private final RecentReject[] rejects = new RecentReject[RECENT_REJECT_SLOTS];
        private RecentReject last;

        private boolean contains(
                GgufSlot<?, ?> slot,
                GGUFModel model,
                GgufKey key,
                long maxBytes,
                long version) {
            if (rejectMatches(last, slot, model, key, maxBytes, version)) {
                return true;
            }
            if (last != null && rejectExpired(last)) {
                last = null;
            }
            int index = index(slot, model, key, maxBytes);
            RecentReject reject = rejects[index];
            if (rejectMatches(reject, slot, model, key, maxBytes, version)) {
                last = reject;
                return true;
            }
            if (reject != null && (rejectExpired(reject) || reject.version() != version)) {
                rejects[index] = null;
            }
            return false;
        }

        private void put(
                GgufSlot<?, ?> slot,
                GGUFModel model,
                GgufKey key,
                long maxBytes,
                long version) {
            RecentReject reject = new RecentReject(slot, new WeakReference<>(model), key, maxBytes, version);
            last = reject;
            rejects[index(slot, model, key, maxBytes)] = reject;
        }

        private int fastSize() {
            if (last == null) {
                return 0;
            }
            if (rejectExpired(last)) {
                last = null;
                return 0;
            }
            return 1;
        }

        private void clear() {
            last = null;
            for (int index = 0; index < rejects.length; index++) {
                rejects[index] = null;
            }
        }

        private static boolean rejectMatches(
                RecentReject reject,
                GgufSlot<?, ?> slot,
                GGUFModel model,
                GgufKey key,
                long maxBytes,
                long version) {
            if (reject == null) {
                return false;
            }
            GGUFModel cachedModel = reject.model().get();
            return cachedModel == model
                    && reject.slot() == slot
                    && reject.key() == key
                    && reject.maxBytes() == maxBytes
                    && reject.version() == version;
        }

        private static boolean rejectExpired(RecentReject reject) {
            return reject.model().get() == null;
        }

        private static int index(GgufSlot<?, ?> slot, GGUFModel model, GgufKey key, long maxBytes) {
            int hash = System.identityHashCode(slot);
            hash = 31 * hash + System.identityHashCode(model);
            hash = 31 * hash + key.hashCode();
            hash = 31 * hash + Long.hashCode(maxBytes);
            return hash & RECENT_REJECT_MASK;
        }
    }

    private record RecentReject(
            GgufSlot<?, ?> slot,
            WeakReference<GGUFModel> model,
            GgufKey key,
            long maxBytes,
            long version) {
    }
}
