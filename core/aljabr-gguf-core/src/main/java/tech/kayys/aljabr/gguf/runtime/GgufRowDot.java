package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufDenseDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufNibRawDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQ8RawDot.*;
import static tech.kayys.aljabr.gguf.runtime.GgufTqRawDot.*;

import tech.kayys.aljabr.gguf.core.GgmlType;

import java.lang.foreign.MemorySegment;

/**
 * Raw GGUF row-dot type dispatch.
 *
 * <p>The public tensor facade validates model/tensor shape and lands here with
 * a concrete row offset. Raw mat-vec fallback also uses this helper directly,
 * so fallback arithmetic no longer needs to depend on {@link GgufTensorOps}.</p>
 * @author bhangun
 */
final class GgufRowDot {
    private GgufRowDot() {
    }

    static float row(
            MemorySegment segment,
            long rowOffset,
            int typeId,
            int columns,
            float[] vector,
            int vectorOffset) {
        return switch (typeFromId(typeId)) {
            case F32 -> dotRowF32(segment, rowOffset, columns, vector, vectorOffset);
            case F16 -> dotRowF16(segment, rowOffset, columns, vector, vectorOffset);
            case BF16 -> dotRowBF16(segment, rowOffset, columns, vector, vectorOffset);
            case Q1_0 -> dotRowQ1_0(segment, rowOffset, columns, vector, vectorOffset);
            case TQ1_0 -> dotRowTQ1_0(segment, rowOffset, columns, vector, vectorOffset);
            case TQ2_0 -> dotRowTQ2_0(segment, rowOffset, columns, vector, vectorOffset);
            case Q4_0 -> dotRowQ4_0(segment, rowOffset, columns, vector, vectorOffset);
            case Q4_1 -> vectorOffset == 0
                    ? GgufRawDot.q4_1(segment, rowOffset, columns, vector, null)
                    : dotRowQ4_1(segment, rowOffset, columns, vector, vectorOffset);
            case Q5_0 -> dotRowQ5_0(segment, rowOffset, columns, vector, vectorOffset);
            case Q5_1 -> vectorOffset == 0
                    ? GgufRawDot.q5_1(segment, rowOffset, columns, vector, null)
                    : dotRowQ5_1(segment, rowOffset, columns, vector, vectorOffset);
            case Q8_0 -> dotRowQ8_0(segment, rowOffset, columns, vector, vectorOffset);
            case Q8_1 -> dotRowQ8_1(segment, rowOffset, columns, vector, vectorOffset);
            case Q8_K -> dotRowQ8K(segment, rowOffset, columns, vector, vectorOffset);
            case MXFP4 -> dotRowMXFP4(segment, rowOffset, columns, vector, vectorOffset);
            case NVFP4 -> dotRowNVFP4(segment, rowOffset, columns, vector, vectorOffset);
            case IQ4_NL -> dotRowIQ4NL(segment, rowOffset, columns, vector, vectorOffset);
            case IQ4_XS -> dotRowIQ4XS(segment, rowOffset, columns, vector, vectorOffset);
            case Q2_K -> vectorOffset == 0
                    ? GgufRawDot.q2K(segment, rowOffset, columns, vector, null)
                    : GgufQ2RawDot.dotRowQ2K(segment, rowOffset, columns, vector, vectorOffset);
            case Q3_K -> GgufQ3RawDot.dotRowQ3K(segment, rowOffset, columns, vector, vectorOffset);
            case Q4_K -> vectorOffset == 0
                    ? GgufRawDot.q4K(segment, rowOffset, columns, vector, null)
                    : GgufQ4RawDot.dotRowQ4K(segment, rowOffset, columns, vector, vectorOffset);
            case Q5_K -> vectorOffset == 0
                    ? GgufRawDot.q5K(segment, rowOffset, columns, vector, null)
                    : GgufQ5RawDot.dotRowQ5K(segment, rowOffset, columns, vector, vectorOffset);
            case Q6_K -> GgufQ6RawDot.dotRowQ6K(segment, rowOffset, columns, vector, vectorOffset);
            default -> throw unsupported(typeId);
        };
    }

    private static GgmlType typeFromId(int typeId) {
        try {
            return GgmlType.fromId(typeId);
        } catch (IllegalArgumentException exception) {
            throw unsupported(typeId, exception);
        }
    }

    private static UnsupportedOperationException unsupported(int typeId) {
        return new UnsupportedOperationException("Unsupported GGUF row dot type id: " + typeId);
    }

    private static UnsupportedOperationException unsupported(int typeId, Exception cause) {
        return new UnsupportedOperationException("Unsupported GGUF row dot type id: " + typeId, cause);
    }
}
