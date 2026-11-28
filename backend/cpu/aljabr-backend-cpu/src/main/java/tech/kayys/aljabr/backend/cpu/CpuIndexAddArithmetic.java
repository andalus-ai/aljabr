package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/**
 * Dtype-specific element operations for CPU indexAdd.
 *
 * <p>The execution loop resolves this strategy once, keeping dtype dispatch
 * outside tensor traversal while retaining the existing storage accessors.
 */
enum CpuIndexAddArithmetic {
    I32 {
        @Override
        void copy(MemorySegment source, long[] dimensions, TensorLayout layout,
                MemorySegment destination, long logicalIndex) {
            CpuIndexAddValueAccess.writeIndexAddInt(destination, logicalIndex,
                    CpuIndexAddValueAccess.readIndexAddInt(source, dimensions, layout, logicalIndex));
        }

        @Override
        void add(MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
                MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
                long sourceIndex, long resultIndex) {
            int update = CpuIndexAddValueAccess.readIndexAddInt(
                    updates, sourceDimensions, sourceLayout, sourceIndex);
            int current = CpuIndexAddValueAccess.readIndexAddInt(
                    result, resultDimensions, resultLayout, resultIndex);
            CpuIndexAddValueAccess.writeIndexAddInt(result, resultIndex, current + update);
        }
    },
    I8 {
        @Override
        void copy(MemorySegment source, long[] dimensions, TensorLayout layout,
                MemorySegment destination, long logicalIndex) {
            CpuIndexAddValueAccess.writeIndexAddByte(destination, logicalIndex,
                    CpuIndexAddValueAccess.readIndexAddByte(source, dimensions, layout, logicalIndex));
        }

        @Override
        void add(MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
                MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
                long sourceIndex, long resultIndex) {
            byte update = CpuIndexAddValueAccess.readIndexAddByte(
                    updates, sourceDimensions, sourceLayout, sourceIndex);
            byte current = CpuIndexAddValueAccess.readIndexAddByte(
                    result, resultDimensions, resultLayout, resultIndex);
            CpuIndexAddValueAccess.writeIndexAddByte(result, resultIndex, (byte) (current + update));
        }
    },
    F32 {
        @Override
        void copy(MemorySegment source, long[] dimensions, TensorLayout layout,
                MemorySegment destination, long logicalIndex) {
            copyFloating(DType.F32, source, dimensions, layout, destination, logicalIndex);
        }

        @Override
        void add(MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
                MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
                long sourceIndex, long resultIndex) {
            addFloating(DType.F32, updates, sourceDimensions, sourceLayout,
                    result, resultDimensions, resultLayout, sourceIndex, resultIndex);
        }
    },
    F16 {
        @Override
        void copy(MemorySegment source, long[] dimensions, TensorLayout layout,
                MemorySegment destination, long logicalIndex) {
            copyFloating(DType.F16, source, dimensions, layout, destination, logicalIndex);
        }

        @Override
        void add(MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
                MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
                long sourceIndex, long resultIndex) {
            addFloating(DType.F16, updates, sourceDimensions, sourceLayout,
                    result, resultDimensions, resultLayout, sourceIndex, resultIndex);
        }
    },
    BF16 {
        @Override
        void copy(MemorySegment source, long[] dimensions, TensorLayout layout,
                MemorySegment destination, long logicalIndex) {
            copyFloating(DType.BF16, source, dimensions, layout, destination, logicalIndex);
        }

        @Override
        void add(MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
                MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
                long sourceIndex, long resultIndex) {
            addFloating(DType.BF16, updates, sourceDimensions, sourceLayout,
                    result, resultDimensions, resultLayout, sourceIndex, resultIndex);
        }
    };

    abstract void copy(MemorySegment source, long[] dimensions, TensorLayout layout,
            MemorySegment destination, long logicalIndex);

    abstract void add(MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
            MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
            long sourceIndex, long resultIndex);

    static CpuIndexAddArithmetic forDType(DType dtype) {
        return switch (dtype) {
            case I32 -> I32;
            case I8, INT8 -> I8;
            case F32 -> F32;
            case F16 -> F16;
            case BF16 -> BF16;
            default -> throw new UnsupportedOperationException(
                    "CPU indexAdd does not support dtype: " + dtype);
        };
    }

    private static void copyFloating(DType dtype, MemorySegment source,
            long[] dimensions, TensorLayout layout, MemorySegment destination, long logicalIndex) {
        float value = CpuIndexAddValueAccess.readIndexAddValue(source, dimensions, layout, logicalIndex, dtype);
        CpuIndexAddValueAccess.writeIndexAddValue(destination, logicalIndex, dtype, value);
    }

    private static void addFloating(DType dtype, MemorySegment updates,
            long[] sourceDimensions, TensorLayout sourceLayout,
            MemorySegment result, long[] resultDimensions, TensorLayout resultLayout,
            long sourceIndex, long resultIndex) {
        float update = CpuIndexAddValueAccess.readIndexAddValue(
                updates, sourceDimensions, sourceLayout, sourceIndex, dtype);
        float current = CpuIndexAddValueAccess.readIndexAddValue(
                result, resultDimensions, resultLayout, resultIndex, dtype);
        CpuIndexAddValueAccess.writeIndexAddValue(result, resultIndex, dtype, current + update);
    }
}
