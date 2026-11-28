package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/**
 * Scalar storage kernels shared by the CPU indexed-add implementation.
 *
 * <p>Logical reads honor the supplied tensor layout. Writes target the
 * destination's logical element offset, which is normally a contiguous
 * output layout. Integer paths deliberately avoid floating-point conversion
 * and retain Java two's-complement overflow semantics.</p>
 */
final class CpuIndexAddElementKernels {

    private CpuIndexAddElementKernels() {
    }

    static void copyElement(
            MemorySegment source,
            long[] sourceDimensions,
            TensorLayout sourceLayout,
            MemorySegment destination,
            long destinationLogicalIndex,
            DType dtype) {
        switch (dtype) {
            case I32 -> writeInt(destination, destinationLogicalIndex,
                    readInt(source, sourceDimensions, sourceLayout, destinationLogicalIndex));
            case I8, INT8 -> writeByte(destination, destinationLogicalIndex,
                    readByte(source, sourceDimensions, sourceLayout, destinationLogicalIndex));
            case F32, F16, BF16 -> writeFloating(destination, destinationLogicalIndex, dtype,
                    readFloating(source, sourceDimensions, sourceLayout, destinationLogicalIndex, dtype));
            default -> throw unsupported(dtype);
        }
    }

    static void accumulateElement(
            MemorySegment updates,
            long[] sourceDimensions,
            TensorLayout sourceLayout,
            long sourceLogicalIndex,
            MemorySegment destination,
            long[] destinationDimensions,
            TensorLayout destinationLayout,
            long destinationLogicalIndex,
            DType dtype) {
        switch (dtype) {
            case I32 -> writeInt(destination, destinationLogicalIndex,
                    readInt(destination, destinationDimensions, destinationLayout, destinationLogicalIndex)
                            + readInt(updates, sourceDimensions, sourceLayout, sourceLogicalIndex));
            case I8, INT8 -> writeByte(destination, destinationLogicalIndex,
                    (byte) (readByte(destination, destinationDimensions, destinationLayout, destinationLogicalIndex)
                            + readByte(updates, sourceDimensions, sourceLayout, sourceLogicalIndex)));
            case F32, F16, BF16 -> writeFloating(destination, destinationLogicalIndex, dtype,
                    readFloating(destination, destinationDimensions, destinationLayout, destinationLogicalIndex, dtype)
                            + readFloating(updates, sourceDimensions, sourceLayout, sourceLogicalIndex, dtype));
            default -> throw unsupported(dtype);
        }
    }

    private static int readInt(
            MemorySegment segment, long[] dimensions, TensorLayout layout, long logicalIndex) {
        long offset = CpuTensorLayoutAccess.logicalElementOffset(dimensions, layout, logicalIndex);
        return segment.get(ValueLayout.JAVA_INT, Math.multiplyExact(offset, Integer.BYTES));
    }

    private static void writeInt(MemorySegment segment, long logicalIndex, int value) {
        segment.set(ValueLayout.JAVA_INT, Math.multiplyExact(logicalIndex, Integer.BYTES), value);
    }

    private static byte readByte(
            MemorySegment segment, long[] dimensions, TensorLayout layout, long logicalIndex) {
        long offset = CpuTensorLayoutAccess.logicalElementOffset(dimensions, layout, logicalIndex);
        return segment.get(ValueLayout.JAVA_BYTE, offset);
    }

    private static void writeByte(MemorySegment segment, long logicalIndex, byte value) {
        segment.set(ValueLayout.JAVA_BYTE, logicalIndex, value);
    }

    private static float readFloating(
            MemorySegment segment, long[] dimensions, TensorLayout layout, long logicalIndex, DType dtype) {
        long offset = CpuTensorLayoutAccess.logicalElementOffset(dimensions, layout, logicalIndex);
        return switch (dtype) {
            case F32 -> segment.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(offset, Float.BYTES));
            case F16 -> CpuFloatingPointConversions.halfToFloat(
                    segment.get(ValueLayout.JAVA_SHORT, Math.multiplyExact(offset, Short.BYTES)));
            case BF16 -> CpuFloatingPointConversions.bfloat16ToFloat(
                    segment.get(ValueLayout.JAVA_SHORT, Math.multiplyExact(offset, Short.BYTES)));
            default -> throw unsupported(dtype);
        };
    }

    private static void writeFloating(
            MemorySegment segment, long logicalIndex, DType dtype, float value) {
        switch (dtype) {
            case F32 -> segment.set(ValueLayout.JAVA_FLOAT, Math.multiplyExact(logicalIndex, Float.BYTES), value);
            case F16 -> segment.set(ValueLayout.JAVA_SHORT, Math.multiplyExact(logicalIndex, Short.BYTES),
                    CpuFloatingPointConversions.floatToHalf(value));
            case BF16 -> segment.set(ValueLayout.JAVA_SHORT, Math.multiplyExact(logicalIndex, Short.BYTES),
                    CpuFloatingPointConversions.floatToBfloat16(value));
            default -> throw unsupported(dtype);
        }
    }

    private static UnsupportedOperationException unsupported(DType dtype) {
        return new UnsupportedOperationException("Unsupported CPU indexAdd dtype: " + dtype);
    }
}
