package tech.kayys.aljabr.backend.cpu.ops;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD kernels for dequantizing block-quantized formats back to FP32.
 * 
 * <p>
 * These operations heavily use the Java Vector API to unpack compressed
 * sub-byte fields and perform block-level scaling.
 */
public final class DequantizeOps {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_PREFERRED;

    private DequantizeOps() {
    }

    /**
     * Dequantizes a Q8_0 tensor into an F32 tensor.
     * 
     * <p>
     * Q8_0 format (per block):
     * <ul>
     * <li>1 x FP16 scale (2 bytes)</li>
     * <li>32 x INT8 weights (32 bytes)</li>
     * </ul>
     * Total: 34 bytes per block.
     * 
     * @param src         quantized source memory segment
     * @param dst         floating-point destination memory segment
     * @param numElements total number of elements
     */
    public static void dequantizeQ8_0(MemorySegment src, MemorySegment dst, long numElements) {
        final int blockSize = 32;
        final int blockBytes = 34;

        long blocks = numElements / blockSize;
        for (long b = 0; b < blocks; b++) {
            long srcOffset = b * blockBytes;
            long dstOffset = b * blockSize * 4L;

            // 1. Read the FP16 scale.
            // Java doesn't have native FP16, so we read as short and convert.
            short fp16Scale = src.get(ValueLayout.JAVA_SHORT, srcOffset);
            float scale = Float16.toFloat(fp16Scale);

            // 2. Read 32 int8 weights
            long weightsOffset = srcOffset + 2;

            // Unpack 32 bytes to 32 floats using SIMD
            // Since max vector length might be smaller than 32 (e.g. 16 or 8),
            // we loop inside the block.
            for (int i = 0; i < blockSize; i += F_SPECIES.length()) {
                float[] tmp = new float[F_SPECIES.length()];
                for (int j = 0; j < F_SPECIES.length() && (i + j) < blockSize; j++) {
                    tmp[j] = src.get(ValueLayout.JAVA_BYTE, weightsOffset + i + j);
                }

                FloatVector floats = FloatVector.fromArray(F_SPECIES, tmp, 0).mul(scale);
                floats.intoMemorySegment(dst, dstOffset + (i * 4L), ByteOrder.nativeOrder());
            }
        }
    }

    /**
     * Dequantizes a Q4_0 tensor into an F32 tensor.
     * 
     * <p>
     * Q4_0 format (per block):
     * <ul>
     * <li>1 x FP16 scale (2 bytes)</li>
     * <li>16 x bytes containing 32 x 4-bit weights (16 bytes)</li>
     * </ul>
     * Total: 18 bytes per block.
     * 
     * @param src         quantized source memory segment
     * @param dst         floating-point destination memory segment
     * @param numElements total number of elements
     */
    public static void dequantizeQ4_0(MemorySegment src, MemorySegment dst, long numElements) {
        final int blockSize = 32;
        final int blockBytes = 18;

        long blocks = numElements / blockSize;
        for (long b = 0; b < blocks; b++) {
            long srcOffset = b * blockBytes;
            long dstOffset = b * blockSize * 4L;

            // 1. Read the FP16 scale.
            short fp16Scale = src.get(ValueLayout.JAVA_SHORT, srcOffset);
            float scale = Float16.toFloat(fp16Scale);

            // 2. Read 16 bytes of weights (32 x 4-bit)
            long weightsOffset = srcOffset + 2;

            // Unpack 4-bit weights (lower nibble, then upper nibble)
            // GGML Q4_0: lower nibble is first element, upper nibble is second.
            // Wait, GGML usually treats lower nibble as the first.
            // The formula is: val = (nibble - 8) * scale
            for (int i = 0; i < 16; i++) {
                byte packed = src.get(ValueLayout.JAVA_BYTE, weightsOffset + i);

                int w0 = (packed & 0x0F) - 8;
                int w1 = ((packed >> 4) & 0x0F) - 8;

                float f0 = w0 * scale;
                float f1 = w1 * scale;

                // For simplicity, we just do scalar here as we just write them out
                // sequentially.
                // In a heavily optimized kernel, we would load 16 bytes into vector registers,
                // do bitwise AND and shifts, subtract 8, multiply by scale.
                dst.set(ValueLayout.JAVA_FLOAT, dstOffset + (i * 2L) * 4L, f0);
                dst.set(ValueLayout.JAVA_FLOAT, dstOffset + ((i * 2L) + 1) * 4L, f1);
            }
        }
    }

    // Minimal FP16 to FP32 converter
    public static class Float16 {
        public static float toFloat(short half) {
            int h = half & 0xFFFF;
            int sign = (h >> 15) & 0x00000001;
            int exp = (h >> 10) & 0x0000001F;
            int mant = h & 0x000003FF;

            if (exp == 0) {
                if (mant == 0) {
                    return Float.intBitsToFloat(sign << 31);
                } else {
                    while ((mant & 0x00000400) == 0) {
                        mant <<= 1;
                        exp--;
                    }
                    exp++;
                    mant &= ~0x00000400;
                }
            } else if (exp == 31) {
                if (mant == 0) {
                    return Float.intBitsToFloat((sign << 31) | 0x7F800000);
                } else {
                    return Float.intBitsToFloat((sign << 31) | 0x7F800000 | (mant << 13));
                }
            }

            exp = exp + (127 - 15);
            mant = mant << 13;

            return Float.intBitsToFloat((sign << 31) | (exp << 23) | mant);
        }

        public static short fromFloat(float f) {
            int bits = Float.floatToIntBits(f);
            int sign = (bits >>> 16) & 0x8000;
            int val = (bits & 0x7fffffff) + 0x1000;

            if (val >= 0x47800000) {
                if ((bits & 0x7fffffff) >= 0x47800000) {
                    if (val < 0x7f800000)
                        return (short) (sign | 0x7c00);
                    return (short) (sign | 0x7c00 | (bits & 0x007fffff) >>> 13);
                }
                return (short) (sign | 0x7bff);
            }
            if (val >= 0x38800000) {
                return (short) (sign | val - 0x38000000 >>> 13);
            }
            if (val < 0x33000000)
                return (short) sign;
            val = (bits & 0x7fffffff) >>> 23;
            return (short) (sign | ((bits & 0x7fffff | 0x800000) + (0x800000 >>> val - 102) >>> 126 - val));
        }
    }
}
