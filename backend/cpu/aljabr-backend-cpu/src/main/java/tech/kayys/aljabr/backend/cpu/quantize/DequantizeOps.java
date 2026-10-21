package tech.kayys.aljabr.backend.cpu.quantize;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * CPU operations for dequantizing block formats like Q8_0 back to FP32.
 */
public final class DequantizeOps {

    private DequantizeOps() {
    }

    /**
     * Dequantizes a Q8_0 memory segment into FP32 format.
     * 
     * @param src         Q8_0 source segment
     * @param dst         FP32 destination segment
     * @param numElements number of elements to dequantize
     */
    public static void dequantizeQ8_0(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / QuantizeOps.Q8_0_BLOCK_ELEMS;

        for (long i = 0; i < numBlocks; i++) {
            long srcOffset = i * QuantizeOps.Q8_0_BLOCK_BYTES;
            long dstOffset = i * QuantizeOps.Q8_0_BLOCK_ELEMS * 4L;

            // Read scale (d)
            short d16 = src.get(ValueLayout.JAVA_SHORT, srcOffset);
            float d = float16ToFloat32(d16);

            // Read and dequantize values
            long qsOffset = srcOffset + 2;
            for (int j = 0; j < QuantizeOps.Q8_0_BLOCK_ELEMS; j++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, qsOffset + j);
                float val = q * d;
                dst.set(ValueLayout.JAVA_FLOAT, dstOffset + j * 4L, val);
            }
        }
    }

    // Basic F16 to F32 conversion utility
    private static float float16ToFloat32(short val) {
        int bits = val & 0xffff;
        int s = (bits >>> 15) & 0x00000001;
        int e = (bits >>> 10) & 0x0000001f;
        int m = bits & 0x000003ff;

        if (e == 0) {
            if (m == 0) {
                return Float.intBitsToFloat(s << 31);
            } else {
                while ((m & 0x00000400) == 0) {
                    m <<= 1;
                    e -= 1;
                }
                e += 1;
                m &= ~0x00000400;
            }
        } else if (e == 31) {
            if (m == 0) {
                return Float.intBitsToFloat((s << 31) | 0x7f800000);
            } else {
                return Float.intBitsToFloat((s << 31) | 0x7f800000 | (m << 13));
            }
        }

        e = e + (127 - 15);
        m = m << 13;
        return Float.intBitsToFloat((s << 31) | (e << 23) | m);
    }
}
