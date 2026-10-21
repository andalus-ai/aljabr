package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.ref.WeakReference;
import java.util.Objects;

/**
 * Model-level GGUF mat-vec dispatch.
 *
 * <p>This class chooses between prepared caches and raw row kernels. Keeping
 * the policy here makes cache admission and fallback routing easier to tune
 * without growing the public tensor facade.</p>
 * @author bhangun
 */
final class GgufMatVec {
    private static final int RECENT_PLAN_SLOTS = 256;
    private static final int RECENT_PLAN_MASK = RECENT_PLAN_SLOTS - 1;
    private static final ThreadLocal<RecentPlans> RECENT_PLANS = ThreadLocal.withInitial(RecentPlans::new);

    private GgufMatVec() {
    }

    static void rows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            float[] output,
            boolean parallel) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        MatVecPlan plan = plan(tensor, vector.length);
        rows(model, tensor, vector, output, checkedAllRowCount(plan.layout().rows()), parallel, plan);
    }

    static void rows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        Objects.requireNonNull(vector, "vector");
        Objects.requireNonNull(output, "output");
        MatVecPlan plan = plan(tensor, vector.length);
        rows(model, tensor, vector, output, rowCount, parallel, plan);
    }

    private static void rows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            MatVecPlan plan) {
        MatrixLayout layout = plan.layout();
        long rows = layout.rows();
        if (rowCount < 0 || rowCount > rows) {
            throw new IllegalArgumentException("Requested row count " + rowCount + " is outside tensor rows " + rows);
        }
        if (output.length < rowCount) {
            throw new IllegalArgumentException(
                    "Output length " + output.length + " is smaller than requested rows " + rowCount);
        }
        int columns = layout.columns();
        if (rowCount == 0) {
            return;
        }
        int typeId = plan.typeId();
        GgufPreparedCachePolicy.Family family = plan.family();
        GgufKey key = plan.key();

        if (family != null) {
            GgufPreparedCachePolicy.CachePolicy policy = family.admissionPolicy(rowCount);
            if (policy != null) {
                long maxBytes = policy.maxBytes();
                if (maxBytes > 0) {
                    if (key == null) {
                        key = GgufKey.from(tensor, columns, rows);
                    }
                    if (GgufPrepOps.rows(
                            model, tensor, key, family, maxBytes, columns, vector, output, rowCount, parallel)) {
                        return;
                    }
                }
            }
        }
        runRaw(model, tensor, key, rows, columns, layout.rowBytes(), plan, vector, output, rowCount, parallel);
    }

    private static int checkedAllRowCount(long rows) {
        if (rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor has too many rows for Java arrays: " + rows);
        }
        return (int) rows;
    }

    private static void runRaw(
            GGUFModel model,
            GGUFTensorInfo tensor,
            GgufKey key,
            long rows,
            int columns,
            long rowBytes,
            MatVecPlan plan,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (key == null && plan.rawRouteUsesEstimateHints()) {
            key = GgufKey.from(tensor, columns, rows);
        }
        GgufRawRoute.rows(
                model,
                tensor,
                key,
                rows,
                columns,
                rowBytes,
                plan.rawRoute(),
                plan.rawSubroute(),
                plan.rawHintBlocks(),
                plan.typeId(),
                vector,
                output,
                rowCount,
                parallel);
    }

    static int recentPlanCacheSize() {
        return RECENT_PLANS.get().size();
    }

    static int recentPlanFastCacheSize() {
        return RECENT_PLANS.get().fastSize();
    }

    static void clearRecentPlanCache() {
        RECENT_PLANS.get().clear();
    }

    private static MatVecPlan plan(GGUFTensorInfo tensor, int vectorLength) {
        RecentPlans recent = RECENT_PLANS.get();
        MatVecPlan plan = recent.get(tensor);
        if (plan != null) {
            checkVectorLength(plan.layout().columns(), vectorLength);
            return plan;
        }
        MatrixLayout layout = checkedLayout(tensor, vectorLength);
        int typeId = tensor.typeId();
        GgufPreparedCachePolicy.Family family = GgufPreparedCachePolicy.preparedMatrixCacheFamily(typeId);
        int rawRouteInfo = GgufRawRoute.routeInfoFor(typeId);
        int rawRoute = GgufRawRoute.routeFromInfo(rawRouteInfo);
        int rawSubroute = GgufRawRoute.subrouteFromInfo(rawRouteInfo);
        boolean usesEstimateHints = GgufRawRoute.routeInfoUsesEstimateHints(rawRouteInfo);
        int rawHintBlocks = rawHintBlocks(usesEstimateHints, rawRoute, layout);
        plan = new MatVecPlan(
                layout,
                keyForPlan(tensor, layout, usesEstimateHints),
                typeId,
                family,
                rawRoute,
                rawSubroute,
                rawHintBlocks,
                usesEstimateHints);
        recent.put(tensor, plan);
        return plan;
    }

    private static int rawHintBlocks(boolean usesEstimateHints, int rawRoute, MatrixLayout layout) {
        if (!usesEstimateHints) {
            return 0;
        }
        return switch (rawRoute) {
            case GgufRawRoute.ROUTE_Q32 -> GgufRawPathHints.totalQ32Blocks(layout.rows(), layout.columns());
            case GgufRawRoute.ROUTE_Q2K, GgufRawRoute.ROUTE_Q4K, GgufRawRoute.ROUTE_Q5K ->
                    GgufRawPathHints.totalKBlocks(layout.rows(), layout.columns());
            default -> 0;
        };
    }

    private static GgufKey keyForPlan(
            GGUFTensorInfo tensor,
            MatrixLayout layout,
            boolean usesEstimateHints) {
        return usesEstimateHints ? GgufKey.from(tensor, layout.columns(), layout.rows()) : null;
    }

    private static void checkVectorLength(int columns, int vectorLength) {
        if (vectorLength < columns) {
            throw new IllegalArgumentException("Vector length " + vectorLength + " is smaller than columns " + columns);
        }
    }

    private record MatVecPlan(
            MatrixLayout layout,
            GgufKey key,
            int typeId,
            GgufPreparedCachePolicy.Family family,
            int rawRoute,
            int rawSubroute,
            int rawHintBlocks,
            boolean rawRouteUsesEstimateHints) {
    }

    private static final class RecentPlans {
        private final RecentPlan[] plans = new RecentPlan[RECENT_PLAN_SLOTS];
        private RecentPlan last;

        private MatVecPlan get(GGUFTensorInfo tensor) {
            MatVecPlan plan = planIfMatches(last, tensor);
            if (plan != null) {
                return plan;
            }
            int index = index(tensor);
            RecentPlan recent = plans[index];
            plan = planIfMatches(recent, tensor);
            if (plan != null) {
                last = recent;
                return recent.plan();
            }
            if (recent != null && recentExpired(recent)) {
                plans[index] = null;
            }
            return null;
        }

        private void put(GGUFTensorInfo tensor, MatVecPlan plan) {
            RecentPlan recent = new RecentPlan(new WeakReference<>(tensor), plan);
            last = recent;
            plans[index(tensor)] = recent;
        }

        private int size() {
            int size = 0;
            for (int index = 0; index < plans.length; index++) {
                RecentPlan recent = plans[index];
                if (recent == null) {
                    continue;
                }
                if (recent.tensor().get() == null) {
                    plans[index] = null;
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
            for (int index = 0; index < plans.length; index++) {
                plans[index] = null;
            }
        }

        private static MatVecPlan planIfMatches(RecentPlan recent, GGUFTensorInfo tensor) {
            if (recent == null) {
                return null;
            }
            GGUFTensorInfo cachedTensor = recent.tensor().get();
            return cachedTensor == tensor ? recent.plan() : null;
        }

        private static boolean recentExpired(RecentPlan recent) {
            return recent.tensor().get() == null;
        }

        private static int index(GGUFTensorInfo tensor) {
            return System.identityHashCode(tensor) & RECENT_PLAN_MASK;
        }
    }

    private record RecentPlan(
            WeakReference<GGUFTensorInfo> tensor,
            MatVecPlan plan) {
    }
}
