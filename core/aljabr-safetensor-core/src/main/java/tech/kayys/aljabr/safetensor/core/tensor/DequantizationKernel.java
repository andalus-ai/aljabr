package tech.kayys.aljabr.safetensor.core.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized dequantization kernels using Java Vector API.
 * <p>
 * Supports INT8 and INT4 (GPTQ-style packed) formats.
 * @author bhangun
 */
public class DequantizationKernel {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_PREFERRED;

    /**
     * The 16 NormalFloat-4 quantization levels.
     */
    private static final float[] NF4_TABLE = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
        -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
        0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
    };

    /**
     * Dequantize INT8 weights to Float32.
     * Symmetric quantization expected: weight = qweight * scale.
     */
    public static void dequantizeInt8(MemorySegment src, MemorySegment dst, MemorySegment scales, long numel) {
        // Assume scales is 1-element (per-tensor) or matches dimensions.
        // For now, implement per-tensor symmetric INT8 for simplicity.
        float scale = scales.get(ValueLayout.JAVA_FLOAT, 0);
        FloatVector vScale = FloatVector.broadcast(F_SPECIES, scale);

        long i = 0;
        for (; i < F_SPECIES.loopBound(numel); i += F_SPECIES.length()) {
            // Load bytes and convert to floats via vectors
            // Note: Efficient byte-to-float vector conversion
            for (int lane = 0; lane < F_SPECIES.length(); lane++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, i + lane);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i + lane, q * scale);
            }
        }
        for (; i < numel; i++) {
            byte q = src.get(ValueLayout.JAVA_BYTE, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, q * scale);
        }
    }

    /**
     * Dequantize INT4 weights (packed 8 per 32-bit word, GPTQ-style).
     * Supports group-wise scaling.
     */
    public static void dequantizeInt4(MemorySegment src, MemorySegment dst, MemorySegment scales, MemorySegment zeros, long numel, int groupSize) {
        // Simplified INT4: 2 values per byte [low nibble, high nibble]
        long i = 0;
        long numBytes = (numel + 1) / 2;

        for (; i < numBytes; i++) {
            byte b = src.get(ValueLayout.JAVA_BYTE, i);
            int v1 = b & 0x0F;
            int v2 = (b >> 4) & 0x0F;

            // Apply scales/zeros based on group
            long groupIdx = (i * 2) / groupSize;
            float scale = scales.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx);
            float zero = (zeros != null) ? zeros.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx) : 8.0f;

            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2, (v1 - zero) * scale);
            if (i * 2 + 1 < numel) {
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2 + 1, (v2 - zero) * scale);
            }
        }
    }

    /**
     * Dequantize NF4 weights (BitsAndBytes NormalFloat-4).
     */
    public static void dequantizeNf4(MemorySegment src, MemorySegment dst, MemorySegment scales, long numel, int groupSize) {
        long i = 0;
        long numBytes = (numel + 1) / 2;

        for (; i < numBytes; i++) {
            byte b = src.get(ValueLayout.JAVA_BYTE, i);
            // BnB NF4 extracts high nibble first (for even indices) and low nibble second
            int v1 = (b >> 4) & 0x0F;
            int v2 = b & 0x0F;

            long groupIdx = (i * 2) / groupSize;
            float scale = scales.getAtIndex(ValueLayout.JAVA_FLOAT, groupIdx);

            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2, NF4_TABLE[v1] * scale);
            if (i * 2 + 1 < numel) {
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i * 2 + 1, NF4_TABLE[v2] * scale);
            }
        }
    }

    /**
     * Dequantize F16 weights to Float32.
     */
    public static void dequantizeF16(MemorySegment src, MemorySegment dst, long numel) {
        dequantizeF16(src, 0L, dst, 0L, numel);
    }

    public static void dequantizeF16(MemorySegment src, long srcByteOffset, MemorySegment dst, long dstByteOffset,
            long numel) {
        for (long i = 0; i < numel; i++) {
            short raw = src.get(ValueLayout.JAVA_SHORT, srcByteOffset + i * Short.BYTES);
            dst.set(ValueLayout.JAVA_FLOAT, dstByteOffset + i * Float.BYTES, float16ToFloat32(raw));
        }
    }

    /**
     * Dequantize BF16 weights to Float32.
     */
    public static void dequantizeBf16(MemorySegment src, MemorySegment dst, long numel) {
        dequantizeBf16(src, 0L, dst, 0L, numel);
    }

    public static void dequantizeBf16(MemorySegment src, long srcByteOffset, MemorySegment dst, long dstByteOffset,
            long numel) {
        // Scalar loop — safe on all platforms including Apple Silicon (128-bit SIMD).
        // BF16 -> F32: shift the 16-bit BF16 bits up by 16 positions to make a valid F32.
        for (long i = 0; i < numel; i++) {
            short raw = src.get(ValueLayout.JAVA_SHORT, srcByteOffset + i * Short.BYTES);
            dst.set(ValueLayout.JAVA_FLOAT, dstByteOffset + i * Float.BYTES,
                    Float.intBitsToFloat((raw & 0xFFFF) << 16));
        }
    }

    /**
     * Transcodes BF16 storage to IEEE F16 storage for Apple MPS mixed matmul.
     * MPS accepts Float32 x Float16 -> Float32 but rejects BF16 in this API.
     */
    public static void transcodeBf16ToF16(MemorySegment src, MemorySegment dst, long numel) {
        for (long i = 0; i < numel; i++) {
            dst.setAtIndex(ValueLayout.JAVA_SHORT, i, bf16ToFloat16(src.getAtIndex(ValueLayout.JAVA_SHORT, i)));
        }
    }

    /**
     * Transcodes a packed F32 segment into F16, element by element.
     * Used to convert dequantized INT4/BNB weights into F16 so they can be
     * consumed directly by the Metal matvec kernel.
     */
    public static void transcodeF32ToF16(MemorySegment src, MemorySegment dst, long numel) {
        for (long i = 0; i < numel; i++) {
            float v = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_SHORT, i, float32ToFloat16(v));
        }
    }

    public static short bf16ToFloat16Value(short bf16) {
        return bf16ToFloat16(bf16);
    }

    private static short bf16ToFloat16(short bf16) {
        int bits = bf16 & 0xFFFF;
        int sign = bits & 0x8000;
        int exponent = (bits >>> 7) & 0xFF;
        int mantissa = bits & 0x7F;

        if (exponent == 0) {
            // BF16 subnormals are far below F16 range, so they collapse to signed zero.
            return (short) sign;
        }
        if (exponent == 0xFF) {
            int payload = mantissa << 3;
            return (short) (sign | 0x7C00 | (payload == 0 ? 0 : Math.max(1, payload)));
        }

        int halfExponent = exponent - 112; // BF16 bias 127 -> F16 bias 15.
        if (halfExponent >= 0x1F) {
            return (short) (sign | 0x7C00);
        }
        if (halfExponent <= 0) {
            // Normal BF16 values near zero can become F16 subnormals; keep the exact converter here.
            return float32ToFloat16(Float.intBitsToFloat(bits << 16));
        }

        return (short) (sign | (halfExponent << 10) | (mantissa << 3));
    }

    private static float float16ToFloat32(short half) {
        int h = half & 0xFFFF;
        int sign = (h >> 15) & 0x1;
        int exponent = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        int f32;
        if (exponent == 0) {
            if (mantissa == 0) {
                f32 = sign << 31;
            } else {
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= ~0x400;
                f32 = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            f32 = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        }
        return Float.intBitsToFloat(f32);
    }

    private static short float32ToFloat16(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = (bits >>> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;

        if (exponent == 0xFF) {
            if (mantissa == 0) {
                return (short) (sign | 0x7C00);
            }
            return (short) (sign | 0x7C00 | Math.max(1, mantissa >>> 13));
        }

        int halfExponent = exponent - 127 + 15;
        if (halfExponent >= 0x1F) {
            return (short) (sign | 0x7C00);
        }
        if (halfExponent <= 0) {
            if (halfExponent < -10) {
                return (short) sign;
            }
            mantissa |= 0x800000;
            int shift = 14 - halfExponent;
            int halfMantissa = mantissa >>> shift;
            int roundBit = (mantissa >>> (shift - 1)) & 1;
            halfMantissa += roundBit;
            return (short) (sign | halfMantissa);
        }

        int halfMantissa = mantissa >>> 13;
        int roundBits = mantissa & 0x1FFF;
        if (roundBits > 0x1000 || (roundBits == 0x1000 && (halfMantissa & 1) != 0)) {
            halfMantissa++;
            if (halfMantissa == 0x400) {
                halfMantissa = 0;
                halfExponent++;
                if (halfExponent >= 0x1F) {
                    return (short) (sign | 0x7C00);
                }
            }
        }
        return (short) (sign | (halfExponent << 10) | (halfMantissa & 0x3FF));
    }
}
