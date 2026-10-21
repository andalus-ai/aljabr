package tech.kayys.aljabr.gguf.runtime;

/**
 * Lookup tables and small bit decoders used by GGUF quantized tensor kernels.
 *
 * <p>This class owns reusable byte/nibble expansion tables so the hot paths in
 * {@link GgufTensorOps} can stay focused on tensor traversal and arithmetic.</p>
 * @author bhangun
 */
final class GgufQuantTables {
    private static final int NIBBLE_PAIR_VALUES_PER_BYTE = 2;
    private static final int Q1_0_SIGNS_PER_BYTE = 8;
    private static final int[] TQ1_0_POW3 = {1, 3, 9, 27, 81};
    private static final int TQ1_0_TRITS_PER_BYTE = TQ1_0_POW3.length;
    private static final int TQ2_0_LANES_PER_BYTE = 4;

    static final byte[] IQ4_NL_VALUES = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };
    static final byte[] MXFP4_VALUES = {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    static final byte[] IQ4_NL_NIBBLE_PAIRS = buildNibblePairTable(IQ4_NL_VALUES);
    static final byte[] MXFP4_NIBBLE_PAIRS = buildNibblePairTable(MXFP4_VALUES);
    static final byte[] Q1_0_SIGNS = buildQ1_0SignTable();
    static final byte[] TQ1_0_TRITS = buildTQ1_0TritTable();
    static final byte[] TQ2_0_LANES = buildTQ2_0LaneTable();

    private static final float[] E8M0_TO_F32_HALF = buildE8M0ToF32HalfTable();
    private static final float[] UE4M3_TO_F32 = buildUe4m3ToF32Table();
    private static final float[] F16_TO_F32 = buildF16ToF32Table();

    private GgufQuantTables() {
    }

    static int q1_0SignBase(int mask) {
        return (mask & 0xFF) * Q1_0_SIGNS_PER_BYTE;
    }

    static int tq1_0TritBase(int packed) {
        return (packed & 0xFF) * TQ1_0_TRITS_PER_BYTE;
    }

    static int tq2_0LaneBase(int packed) {
        return (packed & 0xFF) * TQ2_0_LANES_PER_BYTE;
    }

    static int nibblePairBase(int packed) {
        return (packed & 0xFF) << 1;
    }

    static float e8m0ToF32Half(int exponent) {
        return E8M0_TO_F32_HALF[exponent & 0xFF];
    }

    static float ue4m3ToF32(int value) {
        return UE4M3_TO_F32[value & 0xFF];
    }

    static float f16ToF32(short bits) {
        return F16_TO_F32[bits & 0xFFFF];
    }

    private static byte[] buildNibblePairTable(byte[] values) {
        byte[] table = new byte[256 * NIBBLE_PAIR_VALUES_PER_BYTE];
        for (int quant = 0; quant < 256; quant++) {
            int base = nibblePairBase(quant);
            table[base] = values[quant & 0x0F];
            table[base + 1] = values[quant >>> 4];
        }
        return table;
    }

    private static byte[] buildQ1_0SignTable() {
        byte[] table = new byte[256 * Q1_0_SIGNS_PER_BYTE];
        for (int mask = 0; mask < 256; mask++) {
            int base = mask * Q1_0_SIGNS_PER_BYTE;
            for (int bit = 0; bit < Q1_0_SIGNS_PER_BYTE; bit++) {
                table[base + bit] = (byte) ((((mask >>> bit) & 1) << 1) - 1);
            }
        }
        return table;
    }

    private static byte[] buildTQ1_0TritTable() {
        byte[] table = new byte[256 * TQ1_0_TRITS_PER_BYTE];
        for (int packed = 0; packed < 256; packed++) {
            int base = packed * TQ1_0_TRITS_PER_BYTE;
            for (int trit = 0; trit < TQ1_0_TRITS_PER_BYTE; trit++) {
                int quant = (packed * TQ1_0_POW3[trit]) & 0xFF;
                table[base + trit] = (byte) (((quant * 3) >>> 8) - 1);
            }
        }
        return table;
    }

    private static byte[] buildTQ2_0LaneTable() {
        byte[] table = new byte[256 * TQ2_0_LANES_PER_BYTE];
        for (int packed = 0; packed < 256; packed++) {
            int base = packed * TQ2_0_LANES_PER_BYTE;
            table[base] = (byte) ((packed & 0x03) - 1);
            table[base + 1] = (byte) (((packed >>> 2) & 0x03) - 1);
            table[base + 2] = (byte) (((packed >>> 4) & 0x03) - 1);
            table[base + 3] = (byte) (((packed >>> 6) & 0x03) - 1);
        }
        return table;
    }

    private static float[] buildE8M0ToF32HalfTable() {
        float[] table = new float[256];
        for (int exponent = 0; exponent < table.length; exponent++) {
            int bits = exponent < 2 ? 0x00200000 << exponent : (exponent - 1) << 23;
            table[exponent] = Float.intBitsToFloat(bits);
        }
        return table;
    }

    private static float[] buildUe4m3ToF32Table() {
        float[] table = new float[256];
        for (int value = 0; value < table.length; value++) {
            if (value == 0 || value == 0x7F) {
                table[value] = 0.0f;
                continue;
            }
            int exponent = (value >>> 3) & 0x0F;
            int mantissa = value & 0x07;
            float raw = exponent == 0
                    ? (float) Math.scalb(mantissa, -9)
                    : (float) Math.scalb(1.0f + mantissa / 8.0f, exponent - 7);
            table[value] = raw * 0.5f;
        }
        return table;
    }

    private static float[] buildF16ToF32Table() {
        float[] table = new float[1 << Short.SIZE];
        for (int bits = 0; bits < table.length; bits++) {
            table[bits] = f16ToF32Slow((short) bits);
        }
        return table;
    }

    private static float f16ToF32Slow(short bits) {
        int half = bits & 0xFFFF;
        int sign = (half >>> 15) & 0x1;
        int exponent = (half >>> 10) & 0x1F;
        int mantissa = half & 0x03FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            float value = (float) Math.scalb(mantissa / 1024.0f, -14);
            return sign == 0 ? value : -value;
        }
        if (exponent == 31) {
            if (mantissa == 0) {
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        int floatBits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        return Float.intBitsToFloat(floatBits);
    }
}
