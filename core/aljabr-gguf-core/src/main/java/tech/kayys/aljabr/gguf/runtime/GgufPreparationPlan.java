package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufPreparedCachePolicy.*;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrixCachePlan;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrixCacheStats;

import java.util.Objects;

/**
 * Prepared-matrix planning and admission policy.
 *
 * <p>This helper decides which GGUF tensors are worth unpacking before
 * generation starts while delegating estimate-cache details to {@link GgufPreparedMatrixEstimator}.</p>
 * @author bhangun
 */
final class GgufPreparationPlan {
    private GgufPreparationPlan() {
    }

    static PreparedMatrixCachePlan planPreparedMatrixCaches(GGUFModel model) {
        return planPreparedMatrixCaches(model, model.tensors(), 1);
    }

    static PreparedMatrixCachePlan planPreparedMatrixCaches(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows) {
        return GgufScan.plan(model, tensors, minRows);
    }

    static PreparedMatrixCacheStats prepareMatrixCaches(GGUFModel model) {
        return prepareMatrixCaches(model, model.tensors(), 1);
    }

    static PreparedMatrixCacheStats prepareMatrixCaches(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows) {
        return GgufScan.prepare(model, tensors, minRows);
    }

    static long prepareMatrixCache(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        GgufPreparedCachePolicy.Family family = preparedMatrixCacheFamily(tensor.typeId());
        if (family == null) {
            throw new IllegalArgumentException("Tensor type does not have a prepared matrix cache: " + tensor.name());
        }
        return GgufPreparedMatrixStore.matrixCached(model, tensor, family).estimatedBytes();
    }

    static long estimatePreparedMatrixCacheBytes(GGUFTensorInfo tensor) {
        return GgufPreparedMatrixEstimator.estimate(tensor);
    }

    static long estimatePreparedMatrixCacheBytes(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixEstimator.estimate(model, tensor);
    }

    static int preparedMatrixEstimateCacheSize(GGUFModel model) {
        return GgufPreparedMatrixEstimator.size(model);
    }

    static boolean shouldUsePreparedMatrixCache(
            GGUFTensorInfo tensor,
            int rowCount,
            int minRows,
            long maxBytes) {
        return meetsRowFloor(rowCount, minRows) && admittedBytes(tensor, maxBytes) >= 0L;
    }

    static boolean shouldUsePreparedMatrixCache(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int rowCount,
            int minRows,
            long maxBytes) {
        return meetsRowFloor(rowCount, minRows) && admittedBytes(model, tensor, maxBytes) >= 0L;
    }

    static long preparedMatrixAdmitBytes(GGUFModel model, GGUFTensorInfo tensor, long maxBytes) {
        return admittedBytes(model, tensor, GgufKey.from(tensor), maxBytes);
    }

    static long preparedMatrixAdmitBytes(GGUFModel model, GGUFTensorInfo tensor, GgufKey key, long maxBytes) {
        return admittedBytes(model, tensor, key, maxBytes);
    }

    private static boolean meetsRowFloor(long rowCount, int minRows) {
        return rowCount >= Math.max(1, minRows);
    }

    private static long admittedBytes(GGUFTensorInfo tensor, long maxBytes) {
        if (maxBytes <= 0) {
            return -1L;
        }
        try {
            long estimatedBytes = GgufPreparedMatrixEstimator.estimate(tensor);
            return estimatedBytes <= maxBytes ? estimatedBytes : -1L;
        } catch (RuntimeException ignored) {
            return -1L;
        }
    }

    private static long admittedBytes(GGUFModel model, GGUFTensorInfo tensor, long maxBytes) {
        return admittedBytes(model, tensor, GgufKey.from(tensor), maxBytes);
    }

    private static long admittedBytes(GGUFModel model, GGUFTensorInfo tensor, GgufKey key, long maxBytes) {
        if (maxBytes <= 0) {
            return -1L;
        }
        try {
            long estimatedBytes = GgufPreparedMatrixEstimator.estimate(model, tensor, key);
            return estimatedBytes <= maxBytes ? estimatedBytes : -1L;
        } catch (RuntimeException ignored) {
            return -1L;
        }
    }

    static GgufScanCandidate admitPreparedMatrixCandidate(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int rowFloor,
            long[] reservedCacheBytes) {
        GgufPreparedCachePolicy.Family family = preparedMatrixCacheFamily(tensor.typeId());
        if (family == null) {
            return GgufScanCandidate.unsupported();
        }
        if (!meetsRowFloor(matrixRows(tensor), rowFloor)) {
            return GgufScanCandidate.smallRow();
        }
        long estimatedBytes = GgufPreparedMatrixEstimator.estimate(model, tensor);
        long maxBytes = family.policy().maxBytes();
        if (estimatedBytes > maxBytes
                || !reservePreparedMatrixCacheBytes(family, estimatedBytes, maxBytes, reservedCacheBytes)) {
            return GgufScanCandidate.cacheTooSmall();
        }
        return GgufScanCandidate.ready(estimatedBytes);
    }

}
