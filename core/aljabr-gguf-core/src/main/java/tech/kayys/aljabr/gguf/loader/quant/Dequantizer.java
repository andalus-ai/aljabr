package tech.kayys.aljabr.gguf.loader.quant;

import tech.kayys.aljabr.gguf.core.GGUFTensorInfo;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
/**
 * 
 * Core class for aljabr module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class Dequantizer {
    
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    
    public static float[] dequantize(GGUFTensorInfo t) {
        int n = (int) t.nElements();
        float[] out = new float[n];
        dequantizeInto(t.data(), t.type().id, n, 0, out, 0);
        return out;
    }
    
    public static void dequantizeRow(GGUFTensorInfo t, long row, float[] out) {
        long cols = t.cols();
        long rowBytes = t.type().bytesFor(cols);
        long byteOff = row * rowBytes;
        dequantizeInto(t.data(), t.type().id, (int) cols, byteOff, out, 0);
    }
    
    private static void dequantizeInto(MemorySegment seg, int type, int count, 
                                       long srcOff, float[] dst, int dstOff) {
        tech.kayys.aljabr.gguf.runtime.GgufRowDequantizer.dequantizeRow(seg, srcOff, type, count, dst, dstOff);
    }
    
    private static void dequantF32(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int i = 0; i < n; i++) {
            dst[dOff + i] = seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, off + i * 4L);
        }
    }
    
    private static void dequantF16(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int i = 0; i < n; i++) {
            short bits = seg.get(LE_SHORT, off + i * 2L);
            dst[dOff + i] = f16ToF32(bits);
        }
    }
    
    private static void dequantQ4_0(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 32; b++) {
            long blockOff = off + b * 18L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            int base = dOff + b * 32;
            for (int j = 0; j < 16; j++) {
                int bval = seg.get(ValueLayout.JAVA_BYTE, blockOff + 2 + j) & 0xFF;
                int lo = bval & 0x0F;
                int hi = (bval >> 4) & 0x0F;
                dst[base + j] = (lo - 8) * d;
                dst[base + j + 16] = (hi - 8) * d;
            }
        }
    }
    
    private static void dequantQ4_1(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 32; b++) {
            long blockOff = off + b * 20L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            float m = f16ToF32(seg.get(LE_SHORT, blockOff + 2));
            int base = dOff + b * 32;
            for (int j = 0; j < 16; j++) {
                int bval = seg.get(ValueLayout.JAVA_BYTE, blockOff + 4 + j) & 0xFF;
                int lo = bval & 0x0F;
                int hi = (bval >> 4) & 0x0F;
                dst[base + j] = lo * d + m;
                dst[base + j + 16] = hi * d + m;
            }
        }
    }
    
    private static void dequantQ5_0(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 32; b++) {
            long blockOff = off + b * 22L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            int qh = seg.get(LE_INT, blockOff + 2);
            int base = dOff + b * 32;
            for (int j = 0; j < 16; j++) {
                int bval = seg.get(ValueLayout.JAVA_BYTE, blockOff + 6 + j) & 0xFF;
                int lo = (bval & 0x0F) | (((qh >> j) & 1) << 4);
                int hi = ((bval >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
                dst[base + j] = (lo - 16) * d;
                dst[base + j + 16] = (hi - 16) * d;
            }
        }
    }
    
    private static void dequantQ5_1(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 32; b++) {
            long blockOff = off + b * 24L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            float m = f16ToF32(seg.get(LE_SHORT, blockOff + 2));
            int qh = seg.get(LE_INT, blockOff + 4);
            int base = dOff + b * 32;
            for (int j = 0; j < 16; j++) {
                int bval = seg.get(ValueLayout.JAVA_BYTE, blockOff + 8 + j) & 0xFF;
                int lo = (bval & 0x0F) | (((qh >> j) & 1) << 4);
                int hi = ((bval >> 4) & 0x0F) | (((qh >> (j + 16)) & 1) << 4);
                dst[base + j] = lo * d + m;
                dst[base + j + 16] = hi * d + m;
            }
        }
    }
    
    private static void dequantQ8_0(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 32; b++) {
            long blockOff = off + b * 34L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            int base = dOff + b * 32;
            for (int j = 0; j < 32; j++) {
                byte q = seg.get(ValueLayout.JAVA_BYTE, blockOff + 2 + j);
                dst[base + j] = q * d;
            }
        }
    }
    
    private static void dequantQ2_K(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 256; b++) {
            long blockOff = off + b * 84L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff + 16 + 64));
            float dmin = f16ToF32(seg.get(LE_SHORT, blockOff + 16 + 64 + 2));
            int base = dOff + b * 256;
            
            for (int i = 0; i < 16; i++) {
                int scByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + i) & 0xFF;
                float sc = d * (scByte & 0x0F);
                float min = dmin * (scByte >> 4);
                
                for (int j = 0; j < 16; j++) {
                    int qIdx = i * 16 + j;
                    int qByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + 16 + (qIdx / 4)) & 0xFF;
                    int q = (qByte >> (2 * (qIdx % 4))) & 0x3;
                    dst[base + qIdx] = sc * q - min;
                }
            }
        }
    }
    
    private static void dequantQ3_K(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 256; b++) {
            long blockOff = off + b * 110L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff + 108));
            int base = dOff + b * 256;
            
            for (int i = 0; i < 16; i++) {
                int scIdx = i / 2;
                int scShift = (i & 1) * 6;
                int sc6 = (seg.get(ValueLayout.JAVA_BYTE, blockOff + 96 + scIdx) & 0xFF) >> scShift;
                sc6 = (sc6 & 0x3F) - 32;
                float scale = d * sc6;
                
                for (int j = 0; j < 16; j++) {
                    int eIdx = i * 16 + j;
                    int qByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + 32 + (eIdx / 4)) & 0xFF;
                    int q2 = (qByte >> (2 * (eIdx % 4))) & 0x3;
                    int hmByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + (eIdx / 8)) & 0xFF;
                    int hm = (hmByte >> (eIdx % 8)) & 1;
                    int q3 = q2 - (hm != 0 ? 4 : 0);
                    dst[base + eIdx] = scale * q3;
                }
            }
        }
    }
    
    private static void dequantQ4_K(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 256; b++) {
            long blockOff = off + b * 144L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            float dmin = f16ToF32(seg.get(LE_SHORT, blockOff + 2));
            int base = dOff + b * 256;
            
            for (int i = 0; i < 8; i++) {
                int scPair = seg.get(ValueLayout.JAVA_BYTE, blockOff + 4 + (i * 3 / 2)) & 0xFF;
                int mnPair = seg.get(ValueLayout.JAVA_BYTE, blockOff + 4 + (i * 3 / 2) + 6) & 0xFF;
                int sc6 = (i & 1) == 0 ? (scPair & 0x3F) : ((scPair >> 4) & 0x3F);
                int mn6 = (i & 1) == 0 ? (mnPair & 0x3F) : ((mnPair >> 4) & 0x3F);
                float scale = d * sc6;
                float min = dmin * mn6;
                
                for (int j = 0; j < 16; j++) {
                    int bval = seg.get(ValueLayout.JAVA_BYTE, blockOff + 16 + i * 16 + j) & 0xFF;
                    int lo = bval & 0x0F;
                    int hi = (bval >> 4) & 0x0F;
                    dst[base + i * 32 + j] = scale * lo - min;
                    dst[base + i * 32 + j + 16] = scale * hi - min;
                }
            }
        }
    }
    
    private static void dequantQ5_K(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 256; b++) {
            long blockOff = off + b * 176L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff));
            float dmin = f16ToF32(seg.get(LE_SHORT, blockOff + 2));
            int base = dOff + b * 256;
            
            for (int i = 0; i < 8; i++) {
                int scPair = seg.get(ValueLayout.JAVA_BYTE, blockOff + 4 + (i * 3 / 2)) & 0xFF;
                int mnPair = seg.get(ValueLayout.JAVA_BYTE, blockOff + 4 + (i * 3 / 2) + 6) & 0xFF;
                int sc6 = (i & 1) == 0 ? (scPair & 0x3F) : ((scPair >> 4) & 0x3F);
                int mn6 = (i & 1) == 0 ? (mnPair & 0x3F) : ((mnPair >> 4) & 0x3F);
                float scale = d * sc6;
                float min = dmin * mn6;
                
                for (int j = 0; j < 16; j++) {
                    int bval = seg.get(ValueLayout.JAVA_BYTE, blockOff + 48 + i * 16 + j) & 0xFF;
                    int hmByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + 16 + (i * 16 + j) / 8) & 0xFF;
                    int hmBit0 = (hmByte >> ((i * 32 + j) % 8)) & 1;
                    int hmBit1 = (hmByte >> ((i * 32 + j + 16) % 8)) & 1;
                    int lo = (bval & 0x0F) | (hmBit0 << 4);
                    int hi = ((bval >> 4) & 0x0F) | (hmBit1 << 4);
                    dst[base + i * 32 + j] = scale * lo - min;
                    dst[base + i * 32 + j + 16] = scale * hi - min;
                }
            }
        }
    }
    
    private static void dequantQ6_K(MemorySegment seg, long off, float[] dst, int dOff, int n) {
        for (int b = 0; b < n / 256; b++) {
            long blockOff = off + b * 210L;
            float d = f16ToF32(seg.get(LE_SHORT, blockOff + 208));
            int base = dOff + b * 256;
            
            for (int i = 0; i < 256; i++) {
                int loByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + (i / 2)) & 0xFF;
                int hiByte = seg.get(ValueLayout.JAVA_BYTE, blockOff + 128 + (i / 4)) & 0xFF;
                int lo = (i % 2 == 0) ? (loByte & 0x0F) : ((loByte >> 4) & 0x0F);
                int hi = (hiByte >> (2 * (i % 4))) & 0x3;
                int q = (lo | (hi << 4)) - 32;
                byte sc = seg.get(ValueLayout.JAVA_BYTE, blockOff + 192 + (i / 16));
                dst[base + i] = d * sc * q;
            }
        }
    }
    

    public static float f16ToF32(short bits) {
        int s = (bits >> 15) & 0x1;
        int e = (bits >> 10) & 0x1F;
        int m = bits & 0x3FF;
        
        int fBits;
        if (e == 0) {
            if (m == 0) {
                fBits = s << 31;
            } else {
                while ((m & 0x400) == 0) { m <<= 1; e--; }
                e++;
                m &= ~0x400;
                fBits = (s << 31) | ((e + 112) << 23) | (m << 13);
            }
        } else if (e == 31) {
            fBits = (s << 31) | 0x7F800000 | (m << 13);
        } else {
            fBits = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
        return Float.intBitsToFloat(fBits);
    }
}
