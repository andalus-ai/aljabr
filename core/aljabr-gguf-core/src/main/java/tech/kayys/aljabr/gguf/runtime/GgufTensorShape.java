package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * Shape and row-size validation for GGUF tensors used as matrices.
 *
 * <p>GGUF stores matrix columns in the first shape dimension; this helper owns
 * that convention and the guardrails around Java array-sized operations.</p>
 * @author bhangun
 */
final class GgufTensorShape {
    private static final int RECENT_DIM_SLOTS = 256;
    private static final int RECENT_DIM_MASK = RECENT_DIM_SLOTS - 1;
    private static final ThreadLocal<RecentDims> RECENT_DIMS = ThreadLocal.withInitial(RecentDims::new);
    private static final int[] BLOCK_SIZE_BY_TYPE_ID = blockSizeByTypeId();
    private static final int[] TYPE_SIZE_BY_TYPE_ID = typeSizeByTypeId();
    private static final Map<GGUFTensorInfo, ShapeLayout> SHAPES = new WeakHashMap<>();
    private static final Map<GGUFTensorInfo, MatrixLayout> LAYOUTS = new WeakHashMap<>();
    private static volatile LastShape lastShape = LastShape.empty();
    private static volatile LastLayout lastLayout = LastLayout.empty();

    private GgufTensorShape() {
    }

    static long elementCount(GGUFTensorInfo tensor) {
        long elements = 1;
        for (long dimension : tensor.shape()) {
            elements = Math.multiplyExact(elements, dimension);
        }
        return elements;
    }

    /**
     * GGUF stores matrix columns in shape[0]; the remaining dimensions form rows.
     */
    static long matrixColumns(GGUFTensorInfo tensor) {
        return shape(tensor, RECENT_DIMS.get()).columns();
    }

    static long matrixRows(GGUFTensorInfo tensor) {
        return shape(tensor, RECENT_DIMS.get()).rows();
    }

    static int checkedColumns(GGUFTensorInfo tensor, int vectorLength) {
        return checkedLayout(tensor, vectorLength).columns();
    }

    static MatrixLayout checkedLayout(GGUFTensorInfo tensor, int vectorLength) {
        RecentDims recent = RECENT_DIMS.get();
        MatrixLayout cached = cachedLayout(tensor, recent);
        if (cached != null) {
            checkVectorLength(cached.columns(), vectorLength);
            return cached;
        }
        long columns = shape(tensor, recent).columns();
        if (columns > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor row is too wide for Java arrays: " + columns);
        }
        checkVectorLength(columns, vectorLength);
        return layout(tensor, recent);
    }

    private static void checkVectorLength(long columns, int vectorLength) {
        if (vectorLength < columns) {
            throw new IllegalArgumentException("Vector length " + vectorLength + " is smaller than columns " + columns);
        }
    }

    private static ShapeLayout shape(GGUFTensorInfo tensor, RecentDims recent) {
        ShapeLayout cached = recent.shape(tensor);
        if (cached != null) {
            return cached;
        }
        LastShape last = lastShape;
        GGUFTensorInfo cachedTensor = last.tensor().get();
        if (cachedTensor == tensor) {
            ShapeLayout shape = last.shape();
            recent.rememberShape(tensor, shape);
            return shape;
        }
        if (cachedTensor == null && last.shape() != null) {
            lastShape = LastShape.empty();
        }
        synchronized (SHAPES) {
            ShapeLayout shape = SHAPES.get(tensor);
            if (shape == null) {
                shape = newShape(tensor);
                SHAPES.put(tensor, shape);
            }
            lastShape = new LastShape(new WeakReference<>(tensor), shape);
            recent.rememberShape(tensor, shape);
            return shape;
        }
    }

    private static ShapeLayout newShape(GGUFTensorInfo tensor) {
        long[] shape = tensor.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException("Tensor has no shape: " + tensor.name());
        }
        long rows = 1;
        for (int i = 1; i < shape.length; i++) {
            rows = Math.multiplyExact(rows, shape[i]);
        }
        return new ShapeLayout(shape[0], rows);
    }

    private static MatrixLayout layout(GGUFTensorInfo tensor, RecentDims recent) {
        MatrixLayout cached = cachedLayout(tensor, recent);
        if (cached != null) {
            return cached;
        }
        synchronized (LAYOUTS) {
            MatrixLayout layout = LAYOUTS.get(tensor);
            if (layout == null) {
                layout = newLayout(tensor, recent);
                LAYOUTS.put(tensor, layout);
            }
            lastLayout = new LastLayout(new WeakReference<>(tensor), layout);
            recent.rememberLayout(tensor, layout);
            return layout;
        }
    }

    private static MatrixLayout cachedLayout(GGUFTensorInfo tensor, RecentDims recent) {
        MatrixLayout recentLayout = recent.layout(tensor);
        if (recentLayout != null) {
            return recentLayout;
        }
        LastLayout last = lastLayout;
        GGUFTensorInfo cachedTensor = last.tensor().get();
        MatrixLayout layout = last.layout();
        if (cachedTensor == tensor && layout != null) {
            recent.rememberLayout(tensor, layout);
            return layout;
        }
        if (cachedTensor == null && layout != null) {
            lastLayout = LastLayout.empty();
        }
        return null;
    }

    private static MatrixLayout newLayout(GGUFTensorInfo tensor, RecentDims recent) {
        ShapeLayout shape = shape(tensor, recent);
        long columns = shape.columns();
        if (columns > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor row is too wide for Java arrays: " + columns);
        }
        int cols = (int) columns;
        return new MatrixLayout(cols, shape.rows(), rowByteSize(tensor, cols));
    }

    static void checkRow(GGUFTensorInfo tensor, long row) {
        long rows = matrixRows(tensor);
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException("Row " + row + " is outside tensor row count " + rows);
        }
    }

    static int checkedRows(GGUFTensorInfo tensor) {
        long rows = shape(tensor, RECENT_DIMS.get()).rows();
        if (rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor has too many rows for Java arrays: " + rows);
        }
        return (int) rows;
    }

    static int recentShapeFastCacheSize() {
        return RECENT_DIMS.get().shapeFastSize();
    }

    static int recentLayoutFastCacheSize() {
        return RECENT_DIMS.get().layoutFastSize();
    }

    static void clearRecentDimCache() {
        RECENT_DIMS.get().clear();
    }

    static long rowByteSize(GGUFTensorInfo tensor, int columns) {
        return rowByteSize(tensor.typeId(), columns);
    }

    static long rowByteSize(int typeId, int columns) {
        if (typeId >= 0 && typeId < BLOCK_SIZE_BY_TYPE_ID.length) {
            int blockSize = BLOCK_SIZE_BY_TYPE_ID[typeId];
            int typeSize = TYPE_SIZE_BY_TYPE_ID[typeId];
            if (blockSize > 0 && typeSize > 0 && columns % blockSize == 0) {
                return ((long) columns / blockSize) * typeSize;
            }
        }
        return GgmlType.fromId(typeId).bytesFor(columns);
    }

    record MatrixLayout(int columns, long rows, long rowBytes) {
    }

    private record ShapeLayout(long columns, long rows) {
    }

    private static final class RecentDims {
        private final GGUFTensorInfo[] shapeTensors = new GGUFTensorInfo[RECENT_DIM_SLOTS];
        private final ShapeLayout[] shapes = new ShapeLayout[RECENT_DIM_SLOTS];
        private final GGUFTensorInfo[] layoutTensors = new GGUFTensorInfo[RECENT_DIM_SLOTS];
        private final MatrixLayout[] layouts = new MatrixLayout[RECENT_DIM_SLOTS];
        private GGUFTensorInfo lastShapeTensor;
        private ShapeLayout lastShape;
        private GGUFTensorInfo lastLayoutTensor;
        private MatrixLayout lastLayout;

        private ShapeLayout shape(GGUFTensorInfo tensor) {
            if (lastShapeTensor == tensor) {
                return lastShape;
            }
            int slot = slot(tensor);
            ShapeLayout shape = shapeTensors[slot] == tensor ? shapes[slot] : null;
            if (shape != null) {
                lastShapeTensor = tensor;
                lastShape = shape;
            }
            return shape;
        }

        private void rememberShape(GGUFTensorInfo tensor, ShapeLayout shape) {
            lastShapeTensor = tensor;
            lastShape = shape;
            int slot = slot(tensor);
            shapeTensors[slot] = tensor;
            shapes[slot] = shape;
        }

        private MatrixLayout layout(GGUFTensorInfo tensor) {
            if (lastLayoutTensor == tensor) {
                return lastLayout;
            }
            int slot = slot(tensor);
            MatrixLayout layout = layoutTensors[slot] == tensor ? layouts[slot] : null;
            if (layout != null) {
                lastLayoutTensor = tensor;
                lastLayout = layout;
            }
            return layout;
        }

        private void rememberLayout(GGUFTensorInfo tensor, MatrixLayout layout) {
            lastLayoutTensor = tensor;
            lastLayout = layout;
            int slot = slot(tensor);
            layoutTensors[slot] = tensor;
            layouts[slot] = layout;
        }

        private int shapeFastSize() {
            return lastShapeTensor == null ? 0 : 1;
        }

        private int layoutFastSize() {
            return lastLayoutTensor == null ? 0 : 1;
        }

        private void clear() {
            lastShapeTensor = null;
            lastShape = null;
            lastLayoutTensor = null;
            lastLayout = null;
            for (int index = 0; index < RECENT_DIM_SLOTS; index++) {
                shapeTensors[index] = null;
                shapes[index] = null;
                layoutTensors[index] = null;
                layouts[index] = null;
            }
        }

        private static int slot(GGUFTensorInfo tensor) {
            return System.identityHashCode(tensor) & RECENT_DIM_MASK;
        }
    }

    private record LastShape(WeakReference<GGUFTensorInfo> tensor, ShapeLayout shape) {
        static LastShape empty() {
            return new LastShape(new WeakReference<>(null), null);
        }
    }

    private record LastLayout(WeakReference<GGUFTensorInfo> tensor, MatrixLayout layout) {
        static LastLayout empty() {
            return new LastLayout(new WeakReference<>(null), null);
        }
    }

    private static int[] blockSizeByTypeId() {
        int[] values = new int[maxTypeId() + 1];
        for (GgmlType type : GgmlType.values()) {
            values[type.id] = type.blockSize;
        }
        return values;
    }

    private static int[] typeSizeByTypeId() {
        int[] values = new int[maxTypeId() + 1];
        for (GgmlType type : GgmlType.values()) {
            values[type.id] = type.typeSize;
        }
        return values;
    }

    private static int maxTypeId() {
        int max = 0;
        for (GgmlType type : GgmlType.values()) {
            max = Math.max(max, type.id);
        }
        return max;
    }
}
