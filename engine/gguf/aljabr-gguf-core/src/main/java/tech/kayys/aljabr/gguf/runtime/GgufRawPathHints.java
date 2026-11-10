package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.cachedKHasMins;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.cachedQ32HasBlockBiases;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.rememberKHasMinsHint;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.rememberQ32HasBlockBiasHint;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;

import tech.kayys.aljabr.gguf.loader.GGUFModel;

import java.lang.foreign.MemorySegment;
import java.lang.ref.WeakReference;

/**
 * Cached raw-path metadata hints for choosing GGUF mat-vec reducers.
 *
 * <p>Raw affine K/Q32 formats can skip expensive vector sums when all scanned
 * rows have zero mins/biases. Positive partial scans are safe to cache for the
 * whole tensor; negative hints are cached only after scanning the full matrix.</p>
 * @author bhangun
 */
final class GgufRawPathHints {
    private static final int RECENT_HINT_SLOTS = 256;
    private static final int RECENT_HINT_MASK = RECENT_HINT_SLOTS - 1;
    private static final int Q32_HINT_CODE = -1;
    private static final ThreadLocal<RecentHints> RECENT_HINTS = ThreadLocal.withInitial(RecentHints::new);

    private GgufRawPathHints() {
    }

    static Boolean q2KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return kHasMins(
                model,
                key,
                matrixRows,
                totalKBlocks(matrixRows, columns),
                data,
                rowBytes,
                columns,
                rowCount,
                16,
                GgufKMeta::q2RowsHaveMins);
    }

    static Boolean q4KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return kHasMins(
                model,
                key,
                matrixRows,
                totalKBlocks(matrixRows, columns),
                data,
                rowBytes,
                columns,
                rowCount,
                8,
                GgufKMeta::q4RowsHaveMins);
    }

    static Boolean q5KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return kHasMins(
                model,
                key,
                matrixRows,
                totalKBlocks(matrixRows, columns),
                data,
                rowBytes,
                columns,
                rowCount,
                8,
                GgufKMeta::q5RowsHaveMins);
    }

    static Boolean q2KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int totalBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return kHasMins(
                model,
                key,
                matrixRows,
                totalBlocks,
                data,
                rowBytes,
                columns,
                rowCount,
                16,
                GgufKMeta::q2RowsHaveMins);
    }

    static Boolean q4KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int totalBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return kHasMins(
                model,
                key,
                matrixRows,
                totalBlocks,
                data,
                rowBytes,
                columns,
                rowCount,
                8,
                GgufKMeta::q4RowsHaveMins);
    }

    static Boolean q5KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int totalBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return kHasMins(
                model,
                key,
                matrixRows,
                totalBlocks,
                data,
                rowBytes,
                columns,
                rowCount,
                8,
                GgufKMeta::q5RowsHaveMins);
    }

    private static Boolean q32HasBlockBiasHint(
            GGUFModel model,
            GgufKey key,
            int totalBlocks,
            int q32Route) {
        if (!GgufQ32Route.hasBlockBiases(q32Route)) {
            return null;
        }
        if (key == null) {
            return null;
        }
        RecentHints recent = RECENT_HINTS.get();
        Boolean hint = recent.get(model, key, totalBlocks, Q32_HINT_CODE);
        if (hint != null) {
            return hint;
        }
        hint = cachedQ32HasBlockBiases(model, key, totalBlocks);
        if (hint != null) {
            recent.put(model, key, totalBlocks, Q32_HINT_CODE, hint);
        }
        return hint;
    }

    static Boolean q32HasBlockBias(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount,
            int q32Route) {
        return q32HasBlockBias(
                model,
                key,
                matrixRows,
                totalQ32Blocks(matrixRows, columns),
                data,
                rowBytes,
                columns,
                rowCount,
                q32Route);
    }

    static Boolean q32HasBlockBias(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int totalBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount,
            int q32Route) {
        Boolean hint = q32HasBlockBiasHint(model, key, totalBlocks, q32Route);
        if (hint != null || !GgufQ32Route.hasBlockBiases(q32Route)) {
            return hint;
        }
        boolean hasBlockBiases = switch (q32Route) {
            case GgufQ32Route.Q4_1 -> GgufQ32Meta.q4_1RowsHaveBlockBiases(data, rowBytes, columns, rowCount);
            case GgufQ32Route.Q5_1 -> GgufQ32Meta.q5_1RowsHaveBlockBiases(data, rowBytes, columns, rowCount);
            default -> false;
        };
        if (key != null && (hasBlockBiases || (long) rowCount == matrixRows)) {
            rememberQ32HasBlockBiasHint(model, key, totalBlocks, hasBlockBiases);
            RECENT_HINTS.get().put(model, key, totalBlocks, Q32_HINT_CODE, hasBlockBiases);
        }
        return hasBlockBiases;
    }

    private static Boolean kHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int totalBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount,
            int groupsPerBlock,
            KMinScanner scanner) {
        Boolean hint = kHasMinsHint(model, key, totalBlocks, groupsPerBlock);
        if (hint != null) {
            return hint;
        }
        boolean hasMins = scanner.hasMins(data, rowBytes, columns, rowCount);
        if (key != null && (hasMins || (long) rowCount == matrixRows)) {
            rememberKHasMinsHint(model, key, totalBlocks, groupsPerBlock, hasMins);
            RECENT_HINTS.get().put(model, key, totalBlocks, groupsPerBlock, hasMins);
        }
        return hasMins;
    }

    private static Boolean kHasMinsHint(GGUFModel model, GgufKey key, int totalBlocks, int groupsPerBlock) {
        if (key == null) {
            return null;
        }
        RecentHints recent = RECENT_HINTS.get();
        Boolean hint = recent.get(model, key, totalBlocks, groupsPerBlock);
        if (hint != null) {
            return hint;
        }
        hint = cachedKHasMins(model, key, totalBlocks, groupsPerBlock);
        if (hint != null) {
            recent.put(model, key, totalBlocks, groupsPerBlock, hint);
        }
        return hint;
    }

    static int recentHintCacheSize() {
        return RECENT_HINTS.get().size();
    }

    static int recentHintFastCacheSize() {
        return RECENT_HINTS.get().fastSize();
    }

    static void clearRecentHintCache() {
        RECENT_HINTS.get().clear();
    }

    static int totalKBlocks(long matrixRows, int columns) {
        return Math.toIntExact(Math.multiplyExact(matrixRows, (long) (columns / QK_K)));
    }

    static int totalQ32Blocks(long matrixRows, int columns) {
        return Math.toIntExact(Math.multiplyExact(matrixRows, (long) (columns / Q4_0_BLOCK_SIZE)));
    }

    @FunctionalInterface
    private interface KMinScanner {
        boolean hasMins(MemorySegment data, long rowBytes, int columns, int rowCount);
    }

    private static final class RecentHints {
        private final RecentHint[] hints = new RecentHint[RECENT_HINT_SLOTS];
        private RecentHint last;

        private Boolean get(GGUFModel model, GgufKey key, int totalBlocks, int hintCode) {
            Boolean value = valueIfMatches(last, model, key, totalBlocks, hintCode);
            if (value != null) {
                return value;
            }
            int slot = slot(model, key, totalBlocks, hintCode);
            RecentHint hint = hints[slot];
            value = valueIfMatches(hint, model, key, totalBlocks, hintCode);
            if (value != null) {
                last = hint;
                return value;
            }
            if (hint != null && recentExpired(hint)) {
                hints[slot] = null;
            }
            return null;
        }

        private void put(GGUFModel model, GgufKey key, int totalBlocks, int hintCode, boolean value) {
            RecentHint hint = new RecentHint(new WeakReference<>(model), key, totalBlocks, hintCode, value);
            last = hint;
            hints[slot(model, key, totalBlocks, hintCode)] = hint;
        }

        private int size() {
            int size = 0;
            for (int index = 0; index < hints.length; index++) {
                RecentHint hint = hints[index];
                if (hint == null) {
                    continue;
                }
                if (hint.model().get() == null) {
                    hints[index] = null;
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
            for (int index = 0; index < hints.length; index++) {
                hints[index] = null;
            }
        }

        private static Boolean valueIfMatches(
                RecentHint hint,
                GGUFModel model,
                GgufKey key,
                int totalBlocks,
                int hintCode) {
            if (hint == null) {
                return null;
            }
            GGUFModel cachedModel = hint.model().get();
            if (cachedModel == model
                    && hint.key() == key
                    && hint.totalBlocks() == totalBlocks
                    && hint.hintCode() == hintCode) {
                return hint.value();
            }
            return null;
        }

        private static boolean recentExpired(RecentHint hint) {
            return hint.model().get() == null;
        }

        private static int slot(GGUFModel model, GgufKey key, int totalBlocks, int hintCode) {
            int hash = System.identityHashCode(model);
            hash = 31 * hash + key.hashCode();
            hash = 31 * hash + totalBlocks;
            hash = 31 * hash + hintCode;
            return hash & RECENT_HINT_MASK;
        }
    }

    private record RecentHint(
            WeakReference<GGUFModel> model,
            GgufKey key,
            int totalBlocks,
            int hintCode,
            boolean value) {
    }
}
