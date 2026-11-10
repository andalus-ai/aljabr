package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufPreparedCachePolicy.preparedMatrixCacheBucketCount;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrixCachePlan;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrixCacheStats;

import java.util.Objects;

/**
 * Shared scanner for prepared-matrix cache planning and preparation.
 * @author bhangun
 */
final class GgufScan {
    private int scannedTensors;
    private int matrixTensors;
    private int preparedCandidates;
    private int preparedTensors;
    private int skippedUnsupportedTypeTensors;
    private int skippedSmallRowTensors;
    private int skippedCacheTooSmallTensors;
    private int failedTensors;
    private long estimatedPreparedBytes;
    private long preparedBytes;

    private GgufScan() {
    }

    static PreparedMatrixCachePlan plan(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows) {
        return scan(model, tensors, minRows, false).plan();
    }

    static PreparedMatrixCacheStats prepare(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows) {
        long startNanos = System.nanoTime();
        GgufScan scan = scan(model, tensors, minRows, true);
        return scan.stats(model, System.nanoTime() - startNanos);
    }

    private static GgufScan scan(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows,
            boolean prepare) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensors, "tensors");

        long[] reservedCacheBytes = new long[preparedMatrixCacheBucketCount()];
        int rowFloor = Math.max(1, minRows);
        GgufScan scan = new GgufScan();

        for (GGUFTensorInfo tensor : tensors) {
            scan.scannedTensors++;
            try {
                scan.tryTensor(model, tensor, rowFloor, reservedCacheBytes, prepare);
            } catch (RuntimeException exception) {
                scan.failedTensors++;
            }
        }

        return scan;
    }

    private void tryTensor(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int rowFloor,
            long[] reservedCacheBytes,
            boolean prepare) {
        if (tensor.shape().length < 2) {
            return;
        }
        matrixTensors++;
        GgufScanCandidate candidate =
                GgufPreparationPlan.admitPreparedMatrixCandidate(model, tensor, rowFloor, reservedCacheBytes);
        if (record(candidate) && prepare) {
            preparedBytes += GgufPreparationPlan.prepareMatrixCache(model, tensor);
            preparedTensors++;
        }
    }

    private boolean record(GgufScanCandidate candidate) {
        return switch (candidate.status()) {
            case UNSUPPORTED -> {
                skippedUnsupportedTypeTensors++;
                yield false;
            }
            case SMALL_ROW -> {
                skippedSmallRowTensors++;
                yield false;
            }
            case CACHE_TOO_SMALL -> {
                skippedCacheTooSmallTensors++;
                yield false;
            }
            case READY -> {
                preparedCandidates++;
                estimatedPreparedBytes = Math.addExact(estimatedPreparedBytes, candidate.estimatedBytes());
                yield true;
            }
        };
    }

    private PreparedMatrixCachePlan plan() {
        return new PreparedMatrixCachePlan(
                scannedTensors,
                matrixTensors,
                preparedCandidates,
                skippedUnsupportedTypeTensors,
                skippedSmallRowTensors,
                skippedCacheTooSmallTensors,
                failedTensors,
                estimatedPreparedBytes);
    }

    private PreparedMatrixCacheStats stats(GGUFModel model, long prepareNanos) {
        return new PreparedMatrixCacheStats(
                scannedTensors,
                matrixTensors,
                preparedCandidates,
                preparedTensors,
                skippedUnsupportedTypeTensors,
                skippedSmallRowTensors,
                skippedCacheTooSmallTensors,
                failedTensors,
                preparedBytes,
                GgufTensorOps.preparedMatrixCacheSize(model),
                GgufTensorOps.preparedMatrixCacheBytes(model),
                prepareNanos);
    }
}

/**
 * Admission outcome for a tensor seen by {@link GgufScan}.
 */
enum GgufScanStatus {
    UNSUPPORTED,
    SMALL_ROW,
    CACHE_TOO_SMALL,
    READY
}

/**
 * Prepared-cache candidate metadata returned by the admission policy.
 */
record GgufScanCandidate(GgufScanStatus status, long estimatedBytes) {
    static GgufScanCandidate unsupported() {
        return new GgufScanCandidate(GgufScanStatus.UNSUPPORTED, 0L);
    }

    static GgufScanCandidate smallRow() {
        return new GgufScanCandidate(GgufScanStatus.SMALL_ROW, 0L);
    }

    static GgufScanCandidate cacheTooSmall() {
        return new GgufScanCandidate(GgufScanStatus.CACHE_TOO_SMALL, 0L);
    }

    static GgufScanCandidate ready(long estimatedBytes) {
        return new GgufScanCandidate(GgufScanStatus.READY, estimatedBytes);
    }
}
