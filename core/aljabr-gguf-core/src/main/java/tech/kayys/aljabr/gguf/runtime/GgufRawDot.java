package tech.kayys.aljabr.gguf.runtime;

import java.lang.foreign.MemorySegment;

/**
 * Single-row raw GGUF mat-vec dot dispatch.
 *
 * <p>Raw multi-row execution lives in {@link GgufRawMatVec}; this helper keeps
 * decode-time one-row dispatch in one narrow place so scheduling code does not
 * grow every direct-dot variant.</p>
 * @author bhangun
 */
final class GgufRawDot {
    private GgufRawDot() {
    }

    static float q2K(
            MemorySegment data,
            int columns,
            float[] vector,
            Boolean cachedHasGroupMins) {
        return q2K(data, 0, columns, vector, cachedHasGroupMins);
    }

    static float q2K(
            MemorySegment data,
            int columns,
            float[] vector,
            boolean hasGroupMins) {
        return q2K(data, 0, columns, vector, hasGroupMins);
    }

    static float q2K(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            return q2K(data, rowOffset, columns, vector, cachedHasGroupMins.booleanValue());
        }
        if (!GgufKMeta.q2RowHasMins(data, rowOffset, columns)) {
            return GgufQ2RawDot.dotRowQ2KNoMins(data, rowOffset, columns, vector, 0);
        }
        return GgufQ2RawDot.dotRowQ2K(data, rowOffset, columns, vector, 0);
    }

    static float q2K(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            boolean hasGroupMins) {
        return hasGroupMins
                ? GgufQ2RawDot.dotRowQ2K(data, rowOffset, columns, vector, 0)
                : GgufQ2RawDot.dotRowQ2KNoMins(data, rowOffset, columns, vector, 0);
    }

    static float q3K(MemorySegment data, int columns, float[] vector) {
        return GgufQ3RawDot.dotRowQ3K(data, 0, columns, vector, 0);
    }

    static float q4K(
            MemorySegment data,
            int columns,
            float[] vector,
            Boolean cachedHasGroupMins) {
        return q4K(data, 0, columns, vector, cachedHasGroupMins);
    }

    static float q4K(
            MemorySegment data,
            int columns,
            float[] vector,
            boolean hasGroupMins) {
        return q4K(data, 0, columns, vector, hasGroupMins);
    }

    static float q4K(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            return q4K(data, rowOffset, columns, vector, cachedHasGroupMins.booleanValue());
        }
        if (!GgufKMeta.q4RowHasMins(data, rowOffset, columns)) {
            return GgufQ4RawDot.dotRowQ4KNoMins(data, rowOffset, columns, vector, 0);
        }
        return GgufQ4RawDot.dotRowQ4K(data, rowOffset, columns, vector, 0);
    }

    static float q4K(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            boolean hasGroupMins) {
        return hasGroupMins
                ? GgufQ4RawDot.dotRowQ4K(data, rowOffset, columns, vector, 0)
                : GgufQ4RawDot.dotRowQ4KNoMins(data, rowOffset, columns, vector, 0);
    }

    static float q5K(
            MemorySegment data,
            int columns,
            float[] vector,
            Boolean cachedHasGroupMins) {
        return q5K(data, 0, columns, vector, cachedHasGroupMins);
    }

    static float q5K(
            MemorySegment data,
            int columns,
            float[] vector,
            boolean hasGroupMins) {
        return q5K(data, 0, columns, vector, hasGroupMins);
    }

    static float q5K(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            Boolean cachedHasGroupMins) {
        if (cachedHasGroupMins != null) {
            return q5K(data, rowOffset, columns, vector, cachedHasGroupMins.booleanValue());
        }
        if (!GgufKMeta.q5RowHasMins(data, rowOffset, columns)) {
            return GgufQ5RawDot.dotRowQ5KNoMins(data, rowOffset, columns, vector, 0);
        }
        return GgufQ5RawDot.dotRowQ5K(data, rowOffset, columns, vector, 0);
    }

    static float q5K(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            boolean hasGroupMins) {
        return hasGroupMins
                ? GgufQ5RawDot.dotRowQ5K(data, rowOffset, columns, vector, 0)
                : GgufQ5RawDot.dotRowQ5KNoMins(data, rowOffset, columns, vector, 0);
    }

    static float q6K(MemorySegment data, int columns, float[] vector) {
        return GgufQ6RawDot.dotRowQ6K(data, 0, columns, vector, 0);
    }

    static float q8(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            float[] vector) {
        return switch (route) {
            case GgufQ8Route.Q1_0 -> GgufTqRawDot.dotRowQ1_0(data, 0, columns, vector, 0);
            case GgufQ8Route.TQ1_0 -> GgufTqRawDot.dotRowTQ1_0(data, 0, columns, vector, 0);
            case GgufQ8Route.TQ2_0 -> GgufTqRawDot.dotRowTQ2_0(data, 0, columns, vector, 0);
            case GgufQ8Route.MXFP4 -> GgufNibRawDot.dotRowMXFP4(data, 0, columns, vector, 0);
            case GgufQ8Route.NVFP4 -> GgufNibRawDot.dotRowNVFP4(data, 0, columns, vector, 0);
            case GgufQ8Route.Q8_0 -> GgufQ8RawDot.dotRowQ8_0(data, 0, columns, vector, 0);
            case GgufQ8Route.Q8_1 -> GgufQ8RawDot.dotRowQ8_1(data, 0, columns, vector, 0);
            case GgufQ8Route.Q8_K -> GgufQ8RawDot.dotRowQ8K(data, 0, columns, vector, 0);
            case GgufQ8Route.IQ4_NL -> GgufNibRawDot.dotRowIQ4NL(data, 0, columns, vector, 0);
            case GgufQ8Route.IQ4_XS -> GgufNibRawDot.dotRowIQ4XS(data, 0, columns, vector, 0);
            default -> throw unsupported("Q8-family", typeId);
        };
    }

    static float q32(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            float[] vector,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            return q32(data, route, typeId, columns, vector, cachedHasBlockBiases.booleanValue());
        }
        return switch (route) {
            case GgufQ32Route.Q4_0 -> GgufQ32RawDot.dotRowQ4_0(data, 0, columns, vector, 0);
            case GgufQ32Route.Q4_1 -> q4_1(data, columns, vector, cachedHasBlockBiases);
            case GgufQ32Route.Q5_0 -> GgufQ32RawDot.dotRowQ5_0(data, 0, columns, vector, 0);
            case GgufQ32Route.Q5_1 -> q5_1(data, columns, vector, cachedHasBlockBiases);
            default -> throw unsupported("Q32-family", typeId);
        };
    }

    static float q32(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            float[] vector,
            boolean hasBlockBiases) {
        return switch (route) {
            case GgufQ32Route.Q4_0 -> GgufQ32RawDot.dotRowQ4_0(data, 0, columns, vector, 0);
            case GgufQ32Route.Q4_1 -> q4_1(data, columns, vector, hasBlockBiases);
            case GgufQ32Route.Q5_0 -> GgufQ32RawDot.dotRowQ5_0(data, 0, columns, vector, 0);
            case GgufQ32Route.Q5_1 -> q5_1(data, columns, vector, hasBlockBiases);
            default -> throw unsupported("Q32-family", typeId);
        };
    }

    static float q4_1(
            MemorySegment data,
            int columns,
            float[] vector,
            Boolean cachedHasBlockBiases) {
        return q4_1(data, 0, columns, vector, cachedHasBlockBiases);
    }

    static float q4_1(
            MemorySegment data,
            int columns,
            float[] vector,
            boolean hasBlockBiases) {
        return q4_1(data, 0, columns, vector, hasBlockBiases);
    }

    static float q4_1(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            return q4_1(data, rowOffset, columns, vector, cachedHasBlockBiases.booleanValue());
        }
        if (!GgufQ32Meta.q4_1RowHasBlockBiases(data, rowOffset, columns)) {
            return GgufQ32RawDot.dotRowQ4_1NoBias(data, rowOffset, columns, vector, 0);
        }
        return GgufQ32RawDot.dotRowQ4_1(data, rowOffset, columns, vector, 0);
    }

    static float q4_1(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            boolean hasBlockBiases) {
        return hasBlockBiases
                ? GgufQ32RawDot.dotRowQ4_1(data, rowOffset, columns, vector, 0)
                : GgufQ32RawDot.dotRowQ4_1NoBias(data, rowOffset, columns, vector, 0);
    }

    static float q5_1(
            MemorySegment data,
            int columns,
            float[] vector,
            Boolean cachedHasBlockBiases) {
        return q5_1(data, 0, columns, vector, cachedHasBlockBiases);
    }

    static float q5_1(
            MemorySegment data,
            int columns,
            float[] vector,
            boolean hasBlockBiases) {
        return q5_1(data, 0, columns, vector, hasBlockBiases);
    }

    static float q5_1(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            Boolean cachedHasBlockBiases) {
        if (cachedHasBlockBiases != null) {
            return q5_1(data, rowOffset, columns, vector, cachedHasBlockBiases.booleanValue());
        }
        if (!GgufQ32Meta.q5_1RowHasBlockBiases(data, rowOffset, columns)) {
            return GgufQ32RawDot.dotRowQ5_1NoBias(data, rowOffset, columns, vector, 0);
        }
        return GgufQ32RawDot.dotRowQ5_1(data, rowOffset, columns, vector, 0);
    }

    static float q5_1(
            MemorySegment data,
            long rowOffset,
            int columns,
            float[] vector,
            boolean hasBlockBiases) {
        return hasBlockBiases
                ? GgufQ32RawDot.dotRowQ5_1(data, rowOffset, columns, vector, 0)
                : GgufQ32RawDot.dotRowQ5_1NoBias(data, rowOffset, columns, vector, 0);
    }

    static float dense(
            MemorySegment data,
            int route,
            int typeId,
            int columns,
            float[] vector) {
        return switch (route) {
            case GgufDenseRoute.F32 -> GgufDenseDot.dotRowF32(data, 0, columns, vector, 0);
            case GgufDenseRoute.F16 -> GgufDenseDot.dotRowF16(data, 0, columns, vector, 0);
            case GgufDenseRoute.BF16 -> GgufDenseDot.dotRowBF16(data, 0, columns, vector, 0);
            default -> throw unsupported("dense", typeId);
        };
    }

    static float fallback(
            MemorySegment data,
            int typeId,
            int columns,
            float[] vector) {
        return GgufRowDot.row(data, 0, typeId, columns, vector, 0);
    }

    private static UnsupportedOperationException unsupported(String family, int typeId) {
        return new UnsupportedOperationException("Unsupported GGUF raw " + family + " matvec type id: " + typeId);
    }
}
