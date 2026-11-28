package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/**
 * Typed scalar access for the CPU indexAdd kernel.
 *
 * <p>Reads honor the source tensor's logical layout; writes target the
 * contiguous output allocation. Keeping dtype-specific storage handling here
 * prevents the operation loop from mixing byte offsets and numeric formats.
 */
final class CpuIndexAddValueAccess {

    private CpuIndexAddValueAccess() {
    }

    static int readIndexAddInt(
            MemorySegment segment, long[] dimensions, TensorLayout layout, long logicalIndex) {
        long elementOffset = CpuTensorLayoutAccess.logicalElementOffset(dimensions, layout, logicalIndex);
        return segment.get(ValueLayout.JAVA_INT, Math.multiplyExact(elementOffset, Integer.BYTES));
    }

    static void writeIndexAddInt(MemorySegment segment, long logicalIndex, int value) {
        segment.set(ValueLayout.JAVA_INT, Math.multiplyExact(logicalIndex, Integer.BYTES), value);
    }

    static byte readIndexAddByte(
            MemorySegment segment, long[] dimensions, TensorLayout layout, long logicalIndex) {
        long elementOffset = CpuTensorLayoutAccess.logicalElementOffset(dimensions, layout, logicalIndex);
        return segment.get(ValueLayout.JAVA_BYTE, elementOffset);
    }

    static void writeIndexAddByte(MemorySegment segment, long logicalIndex, byte value) {
        segment.set(ValueLayout.JAVA_BYTE, logicalIndex, value);
    }

    static float readIndexAddValue(
            MemorySegment segment, long[] dimensions, TensorLayout layout, long logicalIndex, DType dtype) {
        long elementOffset = CpuTensorLayoutAccess.logicalElementOffset(dimensions, layout, logicalIndex);
        return switch (dtype) {
            case F32 -> segment.get(ValueLayout.JAVA_FLOAT,
                    Math.multiplyExact(elementOffset, Float.BYTES));
            case F16 -> CpuFloatingPointConversions.halfToFloat(segment.get(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(elementOffset, Short.BYTES)));
            case BF16 -> CpuFloatingPointConversions.bfloat16ToFloat(segment.get(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(elementOffset, Short.BYTES)));
            default -> throw new UnsupportedOperationException("Unsupported CPU indexAdd dtype: " + dtype);
        };
    }

    static void writeIndexAddValue(
            MemorySegment segment, long logicalIndex, DType dtype, float value) {
        switch (dtype) {
            case F32 -> segment.set(ValueLayout.JAVA_FLOAT,
                    Math.multiplyExact(logicalIndex, Float.BYTES), value);
            case F16 -> segment.set(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(logicalIndex, Short.BYTES), CpuFloatingPointConversions.floatToHalf(value));
            case BF16 -> segment.set(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(logicalIndex, Short.BYTES), CpuFloatingPointConversions.floatToBfloat16(value));
            default -> throw new UnsupportedOperationException("Unsupported CPU indexAdd dtype: " + dtype);
        }
    }


}
