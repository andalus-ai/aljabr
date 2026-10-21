package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;

/**
 * GGUF quant format constants and support predicates for the Java runtime.
 *
 * <p>Keep raw block sizes and type-family decisions here so tensor kernels can
 * share one source of truth with profiling and cache planning code.</p>
 * @author bhangun
 */
final class GgufQuantFormats {
    static final int Q4_0_BLOCK_SIZE = 32;
    static final int Q4_0_BLOCK_BYTES = 18;
    static final int Q4_1_BLOCK_BYTES = 20;
    static final int Q5_0_BLOCK_BYTES = 22;
    static final int Q5_1_BLOCK_BYTES = 24;
    static final int Q1_0_BLOCK_SIZE = 128;
    static final int Q1_0_BLOCK_BYTES = 18;
    static final int TQ1_0_BLOCK_SIZE = 256;
    static final int TQ1_0_BLOCK_BYTES = 54;
    static final int TQ1_0_QUANT_BYTES = 48;
    static final int TQ1_0_HIGH_BYTES = 4;
    static final int TQ1_0_SCALE_OFFSET = TQ1_0_QUANT_BYTES + TQ1_0_HIGH_BYTES;
    static final int TQ2_0_BLOCK_SIZE = 256;
    static final int TQ2_0_BLOCK_BYTES = 66;
    static final int TQ2_0_QUANT_BYTES = 64;
    static final int MXFP4_BLOCK_SIZE = 32;
    static final int MXFP4_BLOCK_BYTES = 17;
    static final int NVFP4_BLOCK_SIZE = 64;
    static final int NVFP4_BLOCK_BYTES = 36;
    static final int NVFP4_SUB_BLOCK_SIZE = 16;
    static final int NVFP4_SUB_BLOCKS = NVFP4_BLOCK_SIZE / NVFP4_SUB_BLOCK_SIZE;
    static final int Q8_0_BLOCK_SIZE = 32;
    static final int Q8_0_BLOCK_BYTES = 34;
    static final int Q8_1_BLOCK_SIZE = 32;
    static final int Q8_1_BLOCK_BYTES = 36;
    static final int IQ4_NL_BLOCK_SIZE = 32;
    static final int IQ4_NL_BLOCK_BYTES = 18;
    static final int QK_K = 256;
    static final int QK_GROUP_SIZE = 16;
    static final int QK_GROUPS_PER_BLOCK = QK_K / QK_GROUP_SIZE;
    static final int QK_SUPER_BLOCK_SIZE = 128;
    static final int QK_GROUPS_PER_SUPER_BLOCK = QK_SUPER_BLOCK_SIZE / QK_GROUP_SIZE;
    static final int Q8_K_BLOCK_BYTES = 292;
    static final int IQ4_XS_BLOCK_BYTES = 136;
    static final int IQ4_XS_GROUP_SIZE = 32;
    static final int IQ4_XS_GROUPS = QK_K / IQ4_XS_GROUP_SIZE;
    static final int Q2_K_BLOCK_BYTES = 84;
    static final int Q3_K_BLOCK_BYTES = 110;
    static final int Q4_K_BLOCK_BYTES = 144;
    static final int Q5_K_BLOCK_BYTES = 176;
    static final int Q6_K_BLOCK_BYTES = 210;
    private static final int FLAG_ROW_DOT = 1;
    private static final int FLAG_Q32_PREPARED = 1 << 1;
    private static final int FLAG_PREPARED_MATVEC = 1 << 2;
    private static final int FLAG_Q8_PREPARED = 1 << 3;
    private static final int FLAG_Q32_BLOCK_BIASES = 1 << 4;
    private static final int[] TYPE_FLAGS = typeFlagsById();
    private static final int[] Q8_BLOCK_SIZES = q8BlockSizesById();
    private static final int[] Q8_BLOCK_BYTES = q8BlockBytesById();
    private static final int[] Q32_BLOCK_BYTES = q32BlockBytesById();

    private GgufQuantFormats() {
    }

    static boolean supportsQ32PreparedType(int typeId) {
        return hasFlag(typeId, FLAG_Q32_PREPARED);
    }

    static boolean supportsRowDotType(int typeId) {
        return hasFlag(typeId, FLAG_ROW_DOT);
    }

    static boolean supportsPreparedMatVecType(int typeId) {
        return hasFlag(typeId, FLAG_PREPARED_MATVEC);
    }

    static boolean usesQ8PreparedCache(int typeId) {
        return hasFlag(typeId, FLAG_Q8_PREPARED);
    }

    static boolean q32PreparedHasBlockBiases(int typeId) {
        return hasFlag(typeId, FLAG_Q32_BLOCK_BIASES);
    }

    static int q8BlockSize(int typeId) {
        int blockSize = lookup(Q8_BLOCK_SIZES, typeId);
        if (blockSize > 0) {
            return blockSize;
        }
        throw new IllegalArgumentException("Unsupported Q8 prepared type id: " + typeId);
    }

    static int q8BlockBytes(int typeId) {
        int blockBytes = lookup(Q8_BLOCK_BYTES, typeId);
        if (blockBytes > 0) {
            return blockBytes;
        }
        throw new IllegalArgumentException("Unsupported Q8 prepared type id: " + typeId);
    }

    static int q32BlockBytes(int typeId) {
        int blockBytes = lookup(Q32_BLOCK_BYTES, typeId);
        if (blockBytes > 0) {
            return blockBytes;
        }
        throw new IllegalArgumentException("Unsupported Q32 prepared type id: " + typeId);
    }

    private static boolean hasFlag(int typeId, int flag) {
        return (lookup(TYPE_FLAGS, typeId) & flag) != 0;
    }

    private static int lookup(int[] table, int typeId) {
        return typeId >= 0 && typeId < table.length ? table[typeId] : 0;
    }

    private static int[] typeFlagsById() {
        int[] flags = newTypeTable();
        mark(flags, FLAG_ROW_DOT, GgmlType.F32, GgmlType.F16, GgmlType.BF16);
        mark(flags, FLAG_ROW_DOT | FLAG_Q32_PREPARED | FLAG_PREPARED_MATVEC,
                GgmlType.Q4_0, GgmlType.Q4_1, GgmlType.Q5_0, GgmlType.Q5_1);
        mark(flags, FLAG_Q32_BLOCK_BIASES, GgmlType.Q4_1, GgmlType.Q5_1);
        mark(flags, FLAG_ROW_DOT | FLAG_PREPARED_MATVEC,
                GgmlType.Q2_K, GgmlType.Q3_K, GgmlType.Q4_K, GgmlType.Q5_K, GgmlType.Q6_K);
        mark(flags, FLAG_ROW_DOT | FLAG_PREPARED_MATVEC | FLAG_Q8_PREPARED,
                GgmlType.Q1_0,
                GgmlType.TQ1_0,
                GgmlType.TQ2_0,
                GgmlType.MXFP4,
                GgmlType.NVFP4,
                GgmlType.Q8_0,
                GgmlType.Q8_1,
                GgmlType.Q8_K,
                GgmlType.IQ4_NL,
                GgmlType.IQ4_XS);
        return flags;
    }

    private static int[] q8BlockSizesById() {
        int[] sizes = newTypeTable();
        sizes[GgmlType.Q1_0.id] = Q1_0_BLOCK_SIZE;
        sizes[GgmlType.TQ1_0.id] = TQ1_0_BLOCK_SIZE;
        sizes[GgmlType.TQ2_0.id] = TQ2_0_BLOCK_SIZE;
        sizes[GgmlType.MXFP4.id] = MXFP4_BLOCK_SIZE;
        sizes[GgmlType.NVFP4.id] = NVFP4_SUB_BLOCK_SIZE;
        sizes[GgmlType.Q8_0.id] = Q8_0_BLOCK_SIZE;
        sizes[GgmlType.Q8_1.id] = Q8_1_BLOCK_SIZE;
        sizes[GgmlType.Q8_K.id] = QK_K;
        sizes[GgmlType.IQ4_NL.id] = IQ4_NL_BLOCK_SIZE;
        sizes[GgmlType.IQ4_XS.id] = IQ4_XS_GROUP_SIZE;
        return sizes;
    }

    private static int[] q8BlockBytesById() {
        int[] bytes = newTypeTable();
        bytes[GgmlType.Q1_0.id] = Q1_0_BLOCK_BYTES;
        bytes[GgmlType.TQ1_0.id] = TQ1_0_BLOCK_BYTES;
        bytes[GgmlType.TQ2_0.id] = TQ2_0_BLOCK_BYTES;
        bytes[GgmlType.MXFP4.id] = MXFP4_BLOCK_BYTES;
        bytes[GgmlType.NVFP4.id] = NVFP4_BLOCK_BYTES;
        bytes[GgmlType.Q8_0.id] = Q8_0_BLOCK_BYTES;
        bytes[GgmlType.Q8_1.id] = Q8_1_BLOCK_BYTES;
        bytes[GgmlType.Q8_K.id] = Q8_K_BLOCK_BYTES;
        bytes[GgmlType.IQ4_NL.id] = IQ4_NL_BLOCK_BYTES;
        bytes[GgmlType.IQ4_XS.id] = IQ4_XS_BLOCK_BYTES;
        return bytes;
    }

    private static int[] q32BlockBytesById() {
        int[] bytes = newTypeTable();
        bytes[GgmlType.Q4_0.id] = Q4_0_BLOCK_BYTES;
        bytes[GgmlType.Q4_1.id] = Q4_1_BLOCK_BYTES;
        bytes[GgmlType.Q5_0.id] = Q5_0_BLOCK_BYTES;
        bytes[GgmlType.Q5_1.id] = Q5_1_BLOCK_BYTES;
        return bytes;
    }

    private static int[] newTypeTable() {
        int maxId = 0;
        for (GgmlType type : GgmlType.values()) {
            maxId = Math.max(maxId, type.id);
        }
        return new int[maxId + 1];
    }

    private static void mark(int[] flags, int flag, GgmlType... types) {
        for (GgmlType type : types) {
            flags[type.id] |= flag;
        }
    }
}
