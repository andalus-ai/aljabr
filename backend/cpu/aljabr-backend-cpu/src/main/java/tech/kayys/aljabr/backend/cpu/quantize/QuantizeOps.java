package tech.kayys.aljabr.backend.cpu.quantize;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * CPU operations for quantizing FP32 tensors into block formats like Q8_0.
 */
public final class QuantizeOps {

    // Q8_0 block size is 32 elements.
    // Memory layout per block:
    // float16 d (scale factor) -> 2 bytes
    // int8_t qs[32] (quantized values) -> 32 bytes
    // Total block size: 34 bytes
    public static final int Q8_0_BLOCK_ELEMS = 32;
    public static final int Q8_0_BLOCK_BYTES = 34;

    private QuantizeOps() {
    }

    /**
     * Quantizes an FP32 memory segment into Q8_0 format.
     * 
     * @param src         FP32 source segment
     * @param dst         Q8_0 destination segment
     * @param numElements number of elements to quantize
     */
    public static void quantizeQ8_0(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / Q8_0_BLOCK_ELEMS;

        for (long i = 0; i < numBlocks; i++) {
            long srcOffset = i * Q8_0_BLOCK_ELEMS * 4L;
            long dstOffset = i * Q8_0_BLOCK_BYTES;

            // Find max absolute value in the block
            float maxAbs = 0.0f;
            for (int j = 0; j < Q8_0_BLOCK_ELEMS; j++) {
                float val = src.get(ValueLayout.JAVA_FLOAT, srcOffset + j * 4L);
                maxAbs = Math.max(maxAbs, Math.abs(val));
            }

            // Calculate scale (d)
            float d = maxAbs / 127.0f;
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            // Write scale as float16 (simplified as bits for now, assuming IEEE 754
            // half-precision)
            short d16 = float32ToFloat16(d);
            dst.set(ValueLayout.JAVA_SHORT, dstOffset, d16);

            // Write quantized values
            long qsOffset = dstOffset + 2;
            for (int j = 0; j < Q8_0_BLOCK_ELEMS; j++) {
                float val = src.get(ValueLayout.JAVA_FLOAT, srcOffset + j * 4L);
                byte q = (byte) Math.round(val * id);
                dst.set(ValueLayout.JAVA_BYTE, qsOffset + j, q);
            }
        }
    }

    // Basic F32 to F16 conversion utility
    private static short float32ToFloat16(float val) {
        int bits = Float.floatToIntBits(val);
        int sign = (bits >>> 16) & 0x8000;
        int valBits = bits & 0x7fffffff;
        if (valBits >= 0x47800000) {
            if ((bits & 0x7f800000) == 0x7f800000) {
                if (valBits != 0x7f800000)
                    return (short) (sign | 0x7e00 | (valBits >>> 13));
                return (short) (sign | 0x7c00);
            }
            return (short) (sign | 0x7bff);
        }
        if (valBits >= 0x38800000) {
            return (short) (sign | valBits - 0x38000000 >>> 13);
        }
        if (valBits < 0x33000000)
            return (short) sign;
        valBits = (bits & 0x7fffffff) >>> 23;
        return (short) (sign | ((bits & 0x7fffff | 0x800000) + (0x800000 >>> valBits - 102) >>> 126 - valBits));
    }
}
