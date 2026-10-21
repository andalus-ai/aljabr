package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;

import tech.kayys.aljabr.gguf.core.GgmlType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Row-level GGUF dequantization dispatch.
 *
 * <p>This helper owns the format-to-block traversal for materializing quantized
 * rows as floats. Compact block decoders live in {@link GgufBlockDequantizer};
 * K-family block decoders live in {@link GgufKBlockDequantizer}.</p>
 * @author bhangun
 */
public final class GgufRowDequantizer {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufRowDequantizer() {
    }

    public static void dequantizeRow(
            MemorySegment segment,
            long rowOffset,
            int typeId,
            int columns,
            float[] dst,
            int dstOffset) {
        if (typeId == GgmlType.F32.id) {
            long sourceOffset = rowOffset;
            int out = dstOffset;
            for (int i = 0; i < columns; i++) {
                dst[out++] = segment.get(LE_FLOAT, sourceOffset);
                sourceOffset += Float.BYTES;
            }
            return;
        }
        if (typeId == GgmlType.F16.id) {
            long sourceOffset = rowOffset;
            int out = dstOffset;
            for (int i = 0; i < columns; i++) {
                dst[out++] = f16ToF32(segment.get(LE_SHORT, sourceOffset));
                sourceOffset += Short.BYTES;
            }
            return;
        }
        if (typeId == GgmlType.BF16.id) {
            long sourceOffset = rowOffset;
            int out = dstOffset;
            for (int i = 0; i < columns; i++) {
                int bits = (segment.get(LE_SHORT, sourceOffset) & 0xFFFF) << 16;
                dst[out++] = Float.intBitsToFloat(bits);
                sourceOffset += Short.BYTES;
            }
            return;
        }
        if (typeId == GgmlType.Q1_0.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q1_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeQ1_0Block(segment, blockOffset, dst, out);
                blockOffset += Q1_0_BLOCK_BYTES;
                out += Q1_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.TQ1_0.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / TQ1_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeTQ1_0Block(segment, blockOffset, dst, out);
                blockOffset += TQ1_0_BLOCK_BYTES;
                out += TQ1_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.TQ2_0.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / TQ2_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeTQ2_0Block(segment, blockOffset, dst, out);
                blockOffset += TQ2_0_BLOCK_BYTES;
                out += TQ2_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q4_0.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q4_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeQ4_0Block(segment, blockOffset, dst, out);
                blockOffset += Q4_0_BLOCK_BYTES;
                out += Q4_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q4_1.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q4_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeQ4_1Block(segment, blockOffset, dst, out);
                blockOffset += Q4_1_BLOCK_BYTES;
                out += Q4_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q5_0.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q4_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeQ5_0Block(segment, blockOffset, dst, out);
                blockOffset += Q5_0_BLOCK_BYTES;
                out += Q4_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q5_1.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q4_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeQ5_1Block(segment, blockOffset, dst, out);
                blockOffset += Q5_1_BLOCK_BYTES;
                out += Q4_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q8_0.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q8_0_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
                GgufBlockDequantizer.dequantizeQ8Block(segment, blockOffset + 2, d, dst, out);
                blockOffset += Q8_0_BLOCK_BYTES;
                out += Q8_0_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q8_1.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / Q8_1_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
                GgufBlockDequantizer.dequantizeQ8Block(segment, blockOffset + 4, d, dst, out);
                blockOffset += Q8_1_BLOCK_BYTES;
                out += Q8_1_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.Q8_K.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeQ8KBlock(segment, blockOffset, dst, out);
                blockOffset += Q8_K_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        if (typeId == GgmlType.MXFP4.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / MXFP4_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeMXFP4Block(segment, blockOffset, dst, out);
                blockOffset += MXFP4_BLOCK_BYTES;
                out += MXFP4_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.NVFP4.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / NVFP4_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeNVFP4Block(segment, blockOffset, dst, out);
                blockOffset += NVFP4_BLOCK_BYTES;
                out += NVFP4_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.IQ4_NL.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / IQ4_NL_BLOCK_SIZE;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeIQ4NLBlock(segment, blockOffset, dst, out);
                blockOffset += IQ4_NL_BLOCK_BYTES;
                out += IQ4_NL_BLOCK_SIZE;
            }
            return;
        }
        if (typeId == GgmlType.IQ4_XS.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufBlockDequantizer.dequantizeIQ4XSBlock(segment, blockOffset, dst, out);
                blockOffset += IQ4_XS_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        if (typeId == GgmlType.Q2_K.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufKBlockDequantizer.dequantizeQ2KBlock(segment, blockOffset, dst, out);
                blockOffset += Q2_K_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        if (typeId == GgmlType.Q3_K.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufKBlockDequantizer.dequantizeQ3KBlock(segment, blockOffset, dst, out);
                blockOffset += Q3_K_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        if (typeId == GgmlType.Q4_K.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufKBlockDequantizer.dequantizeQ4KBlock(segment, blockOffset, dst, out);
                blockOffset += Q4_K_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        if (typeId == GgmlType.Q5_K.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufKBlockDequantizer.dequantizeQ5KBlock(segment, blockOffset, dst, out);
                blockOffset += Q5_K_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        if (typeId == GgmlType.Q6_K.id) {
            long blockOffset = rowOffset;
            int out = dstOffset;
            int blocks = columns / QK_K;
            for (int block = 0; block < blocks; block++) {
                GgufKBlockDequantizer.dequantizeQ6KBlock(segment, blockOffset, dst, out);
                blockOffset += Q6_K_BLOCK_BYTES;
                out += QK_K;
            }
            return;
        }
        throw new UnsupportedOperationException("Unsupported GGUF row dequant type id: " + typeId);
    }

}
