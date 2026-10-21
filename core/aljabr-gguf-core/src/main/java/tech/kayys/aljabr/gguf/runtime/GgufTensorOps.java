package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;
import java.util.Locale;
import java.util.Objects;

/**
 * Small active GGUF tensor primitive surface for Java-native generation work.
 *
 * <p>This remains the public facade for row-dot, mat-vec, and prepared matrix
 * operations. Package-private helpers hold format, cache, and scheduling
 * details so this class can keep shrinking toward orchestration plus kernels.</p>
 * @author bhangun
 */
public final class GgufTensorOps {
    private GgufTensorOps() {
    }

    /**
     * Common contract for unpacked prepared matrices admitted to GGUF caches.
     */
    public interface PreparedMatrix {
        long estimatedBytes();
    }

    public record Q4KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            int quantStride,
            int groupStride,
            int noMinKernel,
            int precomputedMinsKernel,
            int directMinsKernel,
            /**
             * Unpacked Q4 values, one unsigned nibble per byte, grouped in the
             * same 32-value scale groups used by Q4_K.
             */
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            boolean hasGroupMins) implements PreparedMatrix {
        static final int ROW_KERNEL_NO_MIN_VECTOR = 1;
        static final int ROW_KERNEL_NO_MIN_SCALAR = 2;
        static final int ROW_KERNEL_PRECOMPUTED_MINS_VECTOR = 3;
        static final int ROW_KERNEL_PRECOMPUTED_MINS_SCALAR = 4;
        static final int ROW_KERNEL_DIRECT_MINS_VECTOR = 5;
        static final int ROW_KERNEL_DIRECT_MINS_SCALAR = 6;

        public Q4KMatrix(
                int columns,
                int rows,
                int blocksPerRow,
                byte[] quants,
                float[] groupScales,
                float[] groupMins,
                boolean hasGroupMins) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    kQuantStride(blocksPerRow),
                    k32GroupStride(blocksPerRow),
                    defaultNoMinKernel(),
                    defaultPrecomputedMinsKernel(),
                    defaultDirectMinsKernel(),
                    quants,
                    groupScales,
                    groupMins,
                    hasGroupMins);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) groupScales.length * Float.BYTES
                    + (long) groupMins.length * Float.BYTES;
        }

        private static int defaultNoMinKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_NO_MIN_VECTOR
                    : ROW_KERNEL_NO_MIN_SCALAR;
        }

        private static int defaultPrecomputedMinsKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_PRECOMPUTED_MINS_VECTOR
                    : ROW_KERNEL_PRECOMPUTED_MINS_SCALAR;
        }

        private static int defaultDirectMinsKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_DIRECT_MINS_VECTOR
                    : ROW_KERNEL_DIRECT_MINS_SCALAR;
        }
    }

    public record Q32Matrix(
            int columns,
            int rows,
            int blocksPerRow,
            int quantStride,
            int noBiasKernel,
            int precomputedBiasKernel,
            int directBiasKernel,
            /**
             * Unpacked 32-value block quants. Symmetric formats store signed
             * values, affine formats store unsigned values.
             */
            byte[] quants,
            float[] blockScales,
            float[] blockBiases,
            boolean hasBlockBiases) implements PreparedMatrix {
        static final int ROW_KERNEL_NO_BIAS_VECTOR = 1;
        static final int ROW_KERNEL_NO_BIAS_SCALAR = 2;
        static final int ROW_KERNEL_PRECOMPUTED_BIAS_VECTOR = 3;
        static final int ROW_KERNEL_PRECOMPUTED_BIAS_SCALAR = 4;
        static final int ROW_KERNEL_DIRECT_BIAS_VECTOR = 5;
        static final int ROW_KERNEL_DIRECT_BIAS_SCALAR = 6;

        public Q32Matrix(
                int columns,
                int rows,
                int blocksPerRow,
                byte[] quants,
                float[] blockScales,
                float[] blockBiases,
                boolean hasBlockBiases) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    q32QuantStride(blocksPerRow),
                    defaultNoBiasKernel(),
                    defaultPrecomputedBiasKernel(),
                    defaultDirectBiasKernel(),
                    quants,
                    blockScales,
                    blockBiases,
                    hasBlockBiases);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) blockScales.length * Float.BYTES
                    + (long) blockBiases.length * Float.BYTES;
        }

        private static int defaultNoBiasKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_NO_BIAS_VECTOR
                    : ROW_KERNEL_NO_BIAS_SCALAR;
        }

        private static int defaultPrecomputedBiasKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_PRECOMPUTED_BIAS_VECTOR
                    : ROW_KERNEL_PRECOMPUTED_BIAS_SCALAR;
        }

        private static int defaultDirectBiasKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_DIRECT_BIAS_VECTOR
                    : ROW_KERNEL_DIRECT_BIAS_SCALAR;
        }
    }

    public record Q2KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            int quantStride,
            int groupStride,
            int noMinKernel,
            int precomputedMinsKernel,
            int directMinsKernel,
            /**
             * Unpacked Q2 values, one unsigned two-bit value per byte, grouped
             * in Q2_K's 16-value scale groups.
             */
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            boolean hasGroupMins) implements PreparedMatrix {
        static final int ROW_KERNEL_NO_MIN_VECTOR = 1;
        static final int ROW_KERNEL_NO_MIN_SCALAR = 2;
        static final int ROW_KERNEL_PRECOMPUTED_MINS_VECTOR = 3;
        static final int ROW_KERNEL_PRECOMPUTED_MINS_SCALAR = 4;
        static final int ROW_KERNEL_DIRECT_MINS_VECTOR = 5;
        static final int ROW_KERNEL_DIRECT_MINS_SCALAR = 6;

        public Q2KMatrix(
                int columns,
                int rows,
                int blocksPerRow,
                byte[] quants,
                float[] groupScales,
                float[] groupMins,
                boolean hasGroupMins) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    kQuantStride(blocksPerRow),
                    k16GroupStride(blocksPerRow),
                    defaultNoMinKernel(),
                    defaultPrecomputedMinsKernel(),
                    defaultDirectMinsKernel(),
                    quants,
                    groupScales,
                    groupMins,
                    hasGroupMins);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) groupScales.length * Float.BYTES
                    + (long) groupMins.length * Float.BYTES;
        }

        private static int defaultNoMinKernel() {
            return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_NO_MIN_VECTOR
                    : ROW_KERNEL_NO_MIN_SCALAR;
        }

        private static int defaultPrecomputedMinsKernel() {
            return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_PRECOMPUTED_MINS_VECTOR
                    : ROW_KERNEL_PRECOMPUTED_MINS_SCALAR;
        }

        private static int defaultDirectMinsKernel() {
            return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_DIRECT_MINS_VECTOR
                    : ROW_KERNEL_DIRECT_MINS_SCALAR;
        }
    }

    public record Q3KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            int quantStride,
            int groupStride,
            int rowKernel,
            /**
             * Unpacked signed Q3 values, one value per byte, grouped in
             * Q3_K's 16-value scale groups.
             */
            byte[] quants,
            float[] groupScales) implements PreparedMatrix {
        static final int ROW_KERNEL_VECTOR = 1;
        static final int ROW_KERNEL_SCALAR = 2;

        public Q3KMatrix(
                int columns,
                int rows,
                int blocksPerRow,
                byte[] quants,
                float[] groupScales) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    kQuantStride(blocksPerRow),
                    k16GroupStride(blocksPerRow),
                    defaultK16NoMinKernel(),
                    quants,
                    groupScales);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length + (long) groupScales.length * Float.BYTES;
        }

        private static int defaultK16NoMinKernel() {
            return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_VECTOR
                    : ROW_KERNEL_SCALAR;
        }
    }

    public record Q5KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            int quantStride,
            int groupStride,
            int noMinKernel,
            int precomputedMinsKernel,
            int directMinsKernel,
            /**
             * Unpacked Q5 values, one unsigned 5-bit value per byte, grouped
             * in Q5_K's 32-value scale groups.
             */
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            boolean hasGroupMins) implements PreparedMatrix {
        static final int ROW_KERNEL_NO_MIN_VECTOR = 1;
        static final int ROW_KERNEL_NO_MIN_SCALAR = 2;
        static final int ROW_KERNEL_PRECOMPUTED_MINS_VECTOR = 3;
        static final int ROW_KERNEL_PRECOMPUTED_MINS_SCALAR = 4;
        static final int ROW_KERNEL_DIRECT_MINS_VECTOR = 5;
        static final int ROW_KERNEL_DIRECT_MINS_SCALAR = 6;

        public Q5KMatrix(
                int columns,
                int rows,
                int blocksPerRow,
                byte[] quants,
                float[] groupScales,
                float[] groupMins,
                boolean hasGroupMins) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    kQuantStride(blocksPerRow),
                    k32GroupStride(blocksPerRow),
                    defaultNoMinKernel(),
                    defaultPrecomputedMinsKernel(),
                    defaultDirectMinsKernel(),
                    quants,
                    groupScales,
                    groupMins,
                    hasGroupMins);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length
                    + (long) groupScales.length * Float.BYTES
                    + (long) groupMins.length * Float.BYTES;
        }

        private static int defaultNoMinKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_NO_MIN_VECTOR
                    : ROW_KERNEL_NO_MIN_SCALAR;
        }

        private static int defaultPrecomputedMinsKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_PRECOMPUTED_MINS_VECTOR
                    : ROW_KERNEL_PRECOMPUTED_MINS_SCALAR;
        }

        private static int defaultDirectMinsKernel() {
            return GgufVectorConfig.Q4_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_DIRECT_MINS_VECTOR
                    : ROW_KERNEL_DIRECT_MINS_SCALAR;
        }
    }

    public record Q6KMatrix(
            int columns,
            int rows,
            int blocksPerRow,
            int quantStride,
            int groupStride,
            int rowKernel,
            /**
             * Unpacked signed Q6 values, one value per byte, grouped in
             * Q6_K's 16-value scale groups.
             */
            byte[] quants,
            float[] groupScales) implements PreparedMatrix {
        static final int ROW_KERNEL_VECTOR = 1;
        static final int ROW_KERNEL_SCALAR = 2;

        public Q6KMatrix(
                int columns,
                int rows,
                int blocksPerRow,
                byte[] quants,
                float[] groupScales) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    kQuantStride(blocksPerRow),
                    k16GroupStride(blocksPerRow),
                    defaultK16NoMinKernel(),
                    quants,
                    groupScales);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length + (long) groupScales.length * Float.BYTES;
        }

        private static int defaultK16NoMinKernel() {
            return GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                    ? ROW_KERNEL_VECTOR
                    : ROW_KERNEL_SCALAR;
        }
    }

    public record Q8Matrix(
            int columns,
            int rows,
            int blocksPerRow,
            int blockSize,
            int quantStride,
            int rowRoute,
            int rowKernel,
            byte[] quants,
            float[] blockScales) implements PreparedMatrix {
        static final int ROW_ROUTE_32 = 1;
        static final int ROW_ROUTE_16 = 2;
        static final int ROW_ROUTE_WIDE = 3;
        static final int ROW_ROUTE_BLOCK = 4;
        static final int ROW_KERNEL_32_VECTOR = 1;
        static final int ROW_KERNEL_32_SCALAR = 2;
        static final int ROW_KERNEL_16_VECTOR = 3;
        static final int ROW_KERNEL_16_SCALAR = 4;
        static final int ROW_KERNEL_WIDE_VECTOR = 5;
        static final int ROW_KERNEL_WIDE_SCALAR = 6;
        static final int ROW_KERNEL_BLOCK_VECTOR = 7;
        static final int ROW_KERNEL_BLOCK_SCALAR = 8;

        public Q8Matrix(
                int columns,
                int rows,
                int blocksPerRow,
                int blockSize,
                byte[] quants,
                float[] blockScales) {
            this(
                    columns,
                    rows,
                    blocksPerRow,
                    blockSize,
                    Math.multiplyExact(blocksPerRow, blockSize),
                    rowRoute(blockSize),
                    rowKernel(blockSize),
                    quants,
                    blockScales);
        }

        @Override
        public long estimatedBytes() {
            return (long) quants.length + (long) blockScales.length * Float.BYTES;
        }

        private static int rowRoute(int blockSize) {
            if (blockSize == GgufQuantFormats.Q4_0_BLOCK_SIZE) {
                return ROW_ROUTE_32;
            }
            if (blockSize == GgufQuantFormats.NVFP4_SUB_BLOCK_SIZE) {
                return ROW_ROUTE_16;
            }
            if (blockSize == GgufQuantFormats.Q1_0_BLOCK_SIZE || blockSize == GgufQuantFormats.QK_K) {
                return ROW_ROUTE_WIDE;
            }
            return ROW_ROUTE_BLOCK;
        }

        private static int rowKernel(int blockSize) {
            return switch (rowRoute(blockSize)) {
                case ROW_ROUTE_32 -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                        ? ROW_KERNEL_32_VECTOR
                        : ROW_KERNEL_32_SCALAR;
                case ROW_ROUTE_16 -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                        ? ROW_KERNEL_16_VECTOR
                        : ROW_KERNEL_16_SCALAR;
                case ROW_ROUTE_WIDE -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                        ? ROW_KERNEL_WIDE_VECTOR
                        : ROW_KERNEL_WIDE_SCALAR;
                default -> GgufVectorConfig.SIGNED_BYTE_DOT_VECTOR_ENABLED
                        ? ROW_KERNEL_BLOCK_VECTOR
                        : ROW_KERNEL_BLOCK_SCALAR;
            };
        }
    }

    private static int kQuantStride(int blocksPerRow) {
        return Math.multiplyExact(blocksPerRow, GgufQuantFormats.QK_K);
    }

    private static int k16GroupStride(int blocksPerRow) {
        return Math.multiplyExact(blocksPerRow, GgufQuantFormats.QK_GROUPS_PER_BLOCK);
    }

    private static int k32GroupStride(int blocksPerRow) {
        return Math.multiplyExact(blocksPerRow, 8);
    }

    private static int q32QuantStride(int blocksPerRow) {
        return Math.multiplyExact(blocksPerRow, GgufQuantFormats.Q4_0_BLOCK_SIZE);
    }

    public record PreparedMatrixCacheStats(
            int scannedTensors,
            int matrixTensors,
            int preparedCandidates,
            int preparedTensors,
            int skippedUnsupportedTypeTensors,
            int skippedSmallRowTensors,
            int skippedCacheTooSmallTensors,
            int failedTensors,
            long preparedBytes,
            long cacheEntries,
            long cacheBytes,
            long prepareNanos) {
        public static PreparedMatrixCacheStats empty() {
            return new PreparedMatrixCacheStats(0, 0, 0, 0, 0, 0, 0, 0, 0L, 0L, 0L, 0L);
        }

        public boolean ready() {
            return preparedCandidates == preparedTensors && failedTensors == 0;
        }

        public double prepareMillis() {
            return prepareNanos / 1_000_000.0d;
        }

        public String compactSummary() {
            return String.format(
                    Locale.ROOT,
                    "prepared=%d/%d, cacheEntries=%d, cacheBytes=%.2fMiB, skippedUnsupported=%d, "
                            + "skippedSmall=%d, skippedCacheTooSmall=%d, failed=%d, prepare=%.3fms",
                    preparedTensors,
                    preparedCandidates,
                    cacheEntries,
                    cacheBytes / 1024.0d / 1024.0d,
                    skippedUnsupportedTypeTensors,
                    skippedSmallRowTensors,
                    skippedCacheTooSmallTensors,
                    failedTensors,
                    prepareMillis());
        }
    }

    public record PreparedMatrixCachePlan(
            int scannedTensors,
            int matrixTensors,
            int preparedCandidates,
            int skippedUnsupportedTypeTensors,
            int skippedSmallRowTensors,
            int skippedCacheTooSmallTensors,
            int failedTensors,
            long estimatedPreparedBytes) {
        public static PreparedMatrixCachePlan empty() {
            return new PreparedMatrixCachePlan(0, 0, 0, 0, 0, 0, 0, 0L);
        }

        public boolean hasCandidates() {
            return preparedCandidates > 0;
        }

        public boolean ready() {
            return hasCandidates() && failedTensors == 0;
        }

        public String compactSummary() {
            return String.format(
                    Locale.ROOT,
                    "candidates=%d/%d, estimatedBytes=%.2fMiB, skippedUnsupported=%d, "
                            + "skippedSmall=%d, skippedCacheTooSmall=%d, failed=%d",
                    preparedCandidates,
                    matrixTensors,
                    estimatedPreparedBytes / 1024.0d / 1024.0d,
                    skippedUnsupportedTypeTensors,
                    skippedSmallRowTensors,
                    skippedCacheTooSmallTensors,
                    failedTensors);
        }
    }

    public static final class Q4KWorkBuffer {
        private float[] vectorGroupSums = new float[0];

        public int vectorGroupSumCapacity() {
            return vectorGroupSums.length;
        }

        public void clear() {
            vectorGroupSums = new float[0];
        }

        float[] vectorGroupSums(int requiredLength) {
            if (vectorGroupSums.length < requiredLength) {
                vectorGroupSums = new float[expandedCapacity(requiredLength)];
            }
            return vectorGroupSums;
        }

        private static int expandedCapacity(int requiredLength) {
            if (requiredLength <= 1) {
                return Math.max(0, requiredLength);
            }
            int floorPowerOfTwo = Integer.highestOneBit(requiredLength - 1);
            return floorPowerOfTwo > (Integer.MAX_VALUE >>> 1) ? requiredLength : floorPowerOfTwo << 1;
        }
    }

    public static GGUFTensorInfo findTensor(GGUFModel model, String name) {
        return GgufTensorData.findTensor(model, name);
    }

    public static MemorySegment tensorData(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufTensorData.tensorData(model, tensor);
    }

    public static long elementCount(GGUFTensorInfo tensor) {
        return GgufTensorShape.elementCount(tensor);
    }

    /**
     * GGUF stores matrix columns in shape[0]; the remaining dimensions form rows.
     */
    public static long matrixColumns(GGUFTensorInfo tensor) {
        return GgufTensorShape.matrixColumns(tensor);
    }

    public static long matrixRows(GGUFTensorInfo tensor) {
        return GgufTensorShape.matrixRows(tensor);
    }

    public static boolean supportsRowDotType(int typeId) {
        return GgufQuantFormats.supportsRowDotType(typeId);
    }

    public static void dequantizeRow(GGUFModel model, GGUFTensorInfo tensor, long row, float[] dst) {
        GgufTensorShape.MatrixLayout layout = checkedLayout(tensor, dst.length);
        checkLayoutRow(layout, row);
        MemorySegment data = tensorData(model, tensor);
        long rowOffset = row * layout.rowBytes();
        GgufRowDequantizer.dequantizeRow(data, rowOffset, tensor.typeId(), layout.columns(), dst, 0);
    }

    public static float dotRow(GGUFModel model, GGUFTensorInfo tensor, long row, float[] vector) {
        GgufTensorShape.MatrixLayout layout = checkedLayout(tensor, vector.length);
        checkLayoutRow(layout, row);
        MemorySegment data = tensorData(model, tensor);
        long rowOffset = row * layout.rowBytes();
        return GgufRowDot.row(data, rowOffset, tensor.typeId(), layout.columns(), vector, 0);
    }

    private static void checkLayoutRow(GgufTensorShape.MatrixLayout layout, long row) {
        if (row < 0 || row >= layout.rows()) {
            throw new IllegalArgumentException("Row " + row + " is outside tensor row count " + layout.rows());
        }
    }

    public static void matVec(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output) {
        GgufMatVec.rows(model, tensor, vector, output, false);
    }

    public static void matVecParallel(GGUFModel model, GGUFTensorInfo tensor, float[] vector, float[] output) {
        GgufMatVec.rows(model, tensor, vector, output, true);
    }

    public static void matVecRows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        GgufMatVec.rows(model, tensor, vector, output, rowCount, parallel);
    }

    public static PreparedMatrixCachePlan planPreparedMatrixCaches(GGUFModel model) {
        return GgufPreparationPlan.planPreparedMatrixCaches(model);
    }

    public static PreparedMatrixCachePlan planPreparedMatrixCaches(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows) {
        return GgufPreparationPlan.planPreparedMatrixCaches(model, tensors, minRows);
    }

    public static PreparedMatrixCacheStats prepareMatrixCaches(GGUFModel model) {
        return GgufPreparationPlan.prepareMatrixCaches(model);
    }

    public static PreparedMatrixCacheStats prepareMatrixCaches(
            GGUFModel model,
            Iterable<GGUFTensorInfo> tensors,
            int minRows) {
        return GgufPreparationPlan.prepareMatrixCaches(model, tensors, minRows);
    }

    public static long prepareMatrixCache(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparationPlan.prepareMatrixCache(model, tensor);
    }

    public static long estimatePreparedMatrixCacheBytes(GGUFTensorInfo tensor) {
        return GgufPreparationPlan.estimatePreparedMatrixCacheBytes(tensor);
    }

    public static long estimatePreparedMatrixCacheBytes(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparationPlan.estimatePreparedMatrixCacheBytes(model, tensor);
    }

    static int preparedMatrixEstimateCacheSize(GGUFModel model) {
        return GgufPreparationPlan.preparedMatrixEstimateCacheSize(model);
    }

    public static int clearPreparedMatrixCaches(GGUFModel model) {
        return GgufPreparedMatrixStore.clearPreparedMatrixCaches(model);
    }

    public static int preparedMatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.preparedMatrixCacheSize(model);
    }

    public static long preparedMatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.preparedMatrixCacheBytes(model);
    }

    public static Q32Matrix q32MatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q32MatrixCached(model, tensor);
    }

    public static Q4KMatrix q4KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q4KMatrixCached(model, tensor);
    }

    public static Q2KMatrix q2KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q2KMatrixCached(model, tensor);
    }

    public static Q3KMatrix q3KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q3KMatrixCached(model, tensor);
    }

    public static Q5KMatrix q5KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q5KMatrixCached(model, tensor);
    }

    public static Q6KMatrix q6KMatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q6KMatrixCached(model, tensor);
    }

    public static Q8Matrix q8MatrixCached(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufPreparedMatrixStore.q8MatrixCached(model, tensor);
    }

    public static int clearQ2KMatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q2K);
    }

    public static int clearQ3KMatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q3K);
    }

    public static int clearQ4KMatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q4K);
    }

    public static int clearQ32MatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q32);
    }

    public static int clearQ5KMatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q5K);
    }

    public static int clearQ6KMatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q6K);
    }

    public static int clearQ8MatrixCache(GGUFModel model) {
        return GgufPreparedMatrixStore.clearMatrixCache(model, GgufPreparedCachePolicy.Family.Q8);
    }

    public static int q2KMatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q2K);
    }

    public static int q3KMatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q3K);
    }

    public static int q4KMatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q4K);
    }

    public static int q32MatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q32);
    }

    public static int q5KMatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q5K);
    }

    public static int q6KMatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q6K);
    }

    public static int q8MatrixCacheSize(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheSize(model, GgufPreparedCachePolicy.Family.Q8);
    }

    public static long q2KMatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q2K);
    }

    public static long q3KMatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q3K);
    }

    public static long q4KMatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q4K);
    }

    public static long q32MatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q32);
    }

    public static long q5KMatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q5K);
    }

    public static long q6KMatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q6K);
    }

    public static long q8MatrixCacheBytes(GGUFModel model) {
        return GgufPreparedMatrixStore.matrixCacheBytes(model, GgufPreparedCachePolicy.Family.Q8);
    }

    static boolean shouldUsePreparedMatrixCache(
            GGUFTensorInfo tensor,
            int rowCount,
            int minRows,
            long maxBytes) {
        return GgufPreparationPlan.shouldUsePreparedMatrixCache(tensor, rowCount, minRows, maxBytes);
    }

    static boolean shouldUsePreparedMatrixCache(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int rowCount,
            int minRows,
            long maxBytes) {
        return GgufPreparationPlan.shouldUsePreparedMatrixCache(model, tensor, rowCount, minRows, maxBytes);
    }

    static int preparedMatrixCacheBucketCount() {
        return GgufPreparedCachePolicy.preparedMatrixCacheBucketCount();
    }

    static int preparedMatrixCacheBucket(int typeId) {
        return GgufPreparedCachePolicy.preparedMatrixCacheBucket(typeId);
    }

    static long preparedMatrixCacheMaxBytes(int typeId) {
        return GgufPreparedCachePolicy.preparedMatrixCacheMaxBytes(typeId);
    }

    static boolean shouldParallelize(boolean requested, int rowCount, int columns) {
        return GgufParallelConfig.shouldParallelize(requested, rowCount, columns);
    }

    static int parallelChunkCount(int rowCount) {
        return GgufParallelConfig.parallelChunkCount(rowCount);
    }

    static int parallelChunkCount(boolean requested, int rowCount, int columns) {
        return GgufParallelConfig.parallelChunkCount(requested, rowCount, columns);
    }

    static boolean byteDotVectorPreferred() {
        return GgufVectorConfig.byteDotVectorPreferred();
    }

    static int byteDotVectorLanes() {
        return GgufVectorConfig.byteDotVectorLanes();
    }

    static int preferredFloatVectorLanes() {
        return GgufVectorConfig.preferredFloatVectorLanes();
    }

    static boolean resolveByteDotVectorEnabled(String configured) {
        return GgufVectorConfig.resolveByteDotVectorEnabled(configured);
    }

    public static Q32Matrix q32Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q32Matrix(model, tensor);
    }

    public static Q2KMatrix q2KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q2KMatrix(model, tensor);
    }

    public static Q3KMatrix q3KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q3KMatrix(model, tensor);
    }

    public static Q4KMatrix q4KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q4KMatrix(model, tensor);
    }

    public static Q5KMatrix q5KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q5KMatrix(model, tensor);
    }

    public static Q6KMatrix q6KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q6KMatrix(model, tensor);
    }

    public static Q8Matrix q8Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufBuild.q8Matrix(model, tensor);
    }

    public static void matVecRows(Q4KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q4K(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(Q2KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q2K(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(Q3KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q3K(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(Q32Matrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q32(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(Q5KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q5K(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(Q6KMatrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q6K(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(Q8Matrix matrix, float[] vector, float[] output, int rowCount, boolean parallel) {
        GgufPrepRows.q8(matrix, vector, output, rowCount, parallel);
    }

    public static void matVecRows(
            Q4KMatrix matrix,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel,
            Q4KWorkBuffer workBuffer) {
        Objects.requireNonNull(workBuffer, "workBuffer");
        GgufPrepRows.q4K(matrix, vector, output, rowCount, parallel, workBuffer);
    }

    static boolean supportsQ32PreparedType(int typeId) {
        return GgufQuantFormats.supportsQ32PreparedType(typeId);
    }

    public static boolean supportsPreparedMatVecType(int typeId) {
        return GgufQuantFormats.supportsPreparedMatVecType(typeId);
    }

}
