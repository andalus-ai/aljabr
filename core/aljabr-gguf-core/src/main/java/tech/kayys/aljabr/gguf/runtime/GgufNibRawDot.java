package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.IQ4_NL_NIBBLE_PAIRS;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.MXFP4_NIBBLE_PAIRS;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.e8m0ToF32Half;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.nibblePairBase;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.ue4m3ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.u8;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_NL_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_NL_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_XS_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.IQ4_XS_GROUP_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.MXFP4_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.MXFP4_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_SUB_BLOCKS;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.NVFP4_SUB_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.iq4XSScalePacked;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw nibble-coded GGUF row-dot kernels.
 *
 * <p>MXFP4, NVFP4, IQ4_NL, and IQ4_XS all decode packed nibbles through small
 * lookup tables before applying per-block or per-group scales.</p>
 * @author bhangun
 */
final class GgufNibRawDot {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufNibRawDot() {
    }

    static float dotRowMXFP4(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / MXFP4_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowMXFP4Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowMXFP4Block(
                    segment,
                    blockOffset + MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + MXFP4_BLOCK_SIZE);
            sum2 += dotRowMXFP4Block(
                    segment,
                    blockOffset + 2L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * MXFP4_BLOCK_SIZE);
            sum3 += dotRowMXFP4Block(
                    segment,
                    blockOffset + 3L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * MXFP4_BLOCK_SIZE);
            blockOffset += 4L * MXFP4_BLOCK_BYTES;
            vectorBase += 4 * MXFP4_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowMXFP4Block(segment, blockOffset, vector, vectorBase);
            blockOffset += MXFP4_BLOCK_BYTES;
            vectorBase += MXFP4_BLOCK_SIZE;
        }
        return sum;
    }

    private static float dotRowMXFP4Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = e8m0ToF32Half(u8(segment, blockOffset));
        long quantsOffset = blockOffset + 1;
        return d * dotRowNibble32(segment, quantsOffset, MXFP4_NIBBLE_PAIRS, vector, vectorBase);
    }

    static void dotRowsMXFP4_4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / MXFP4_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowMXFP4Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowMXFP4Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowMXFP4Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowMXFP4Block(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowMXFP4Block(
                    segment,
                    row0Offset + blockOffset + MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + MXFP4_BLOCK_SIZE);
            row1Sum1 += dotRowMXFP4Block(
                    segment,
                    row1Offset + blockOffset + MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + MXFP4_BLOCK_SIZE);
            row2Sum1 += dotRowMXFP4Block(
                    segment,
                    row2Offset + blockOffset + MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + MXFP4_BLOCK_SIZE);
            row3Sum1 += dotRowMXFP4Block(
                    segment,
                    row3Offset + blockOffset + MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + MXFP4_BLOCK_SIZE);
            row0Sum2 += dotRowMXFP4Block(
                    segment,
                    row0Offset + blockOffset + 2L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * MXFP4_BLOCK_SIZE);
            row1Sum2 += dotRowMXFP4Block(
                    segment,
                    row1Offset + blockOffset + 2L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * MXFP4_BLOCK_SIZE);
            row2Sum2 += dotRowMXFP4Block(
                    segment,
                    row2Offset + blockOffset + 2L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * MXFP4_BLOCK_SIZE);
            row3Sum2 += dotRowMXFP4Block(
                    segment,
                    row3Offset + blockOffset + 2L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * MXFP4_BLOCK_SIZE);
            row0Sum3 += dotRowMXFP4Block(
                    segment,
                    row0Offset + blockOffset + 3L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * MXFP4_BLOCK_SIZE);
            row1Sum3 += dotRowMXFP4Block(
                    segment,
                    row1Offset + blockOffset + 3L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * MXFP4_BLOCK_SIZE);
            row2Sum3 += dotRowMXFP4Block(
                    segment,
                    row2Offset + blockOffset + 3L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * MXFP4_BLOCK_SIZE);
            row3Sum3 += dotRowMXFP4Block(
                    segment,
                    row3Offset + blockOffset + 3L * MXFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * MXFP4_BLOCK_SIZE);
            blockOffset += 4L * MXFP4_BLOCK_BYTES;
            vectorBase += 4 * MXFP4_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowMXFP4Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowMXFP4Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowMXFP4Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowMXFP4Block(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += MXFP4_BLOCK_BYTES;
            vectorBase += MXFP4_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowNVFP4(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / NVFP4_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowNVFP4Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowNVFP4Block(
                    segment,
                    blockOffset + NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + NVFP4_BLOCK_SIZE);
            sum2 += dotRowNVFP4Block(
                    segment,
                    blockOffset + 2L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * NVFP4_BLOCK_SIZE);
            sum3 += dotRowNVFP4Block(
                    segment,
                    blockOffset + 3L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * NVFP4_BLOCK_SIZE);
            blockOffset += 4L * NVFP4_BLOCK_BYTES;
            vectorBase += 4 * NVFP4_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowNVFP4Block(segment, blockOffset, vector, vectorBase);
            blockOffset += NVFP4_BLOCK_BYTES;
            vectorBase += NVFP4_BLOCK_SIZE;
        }
        return sum;
    }

    private static float dotRowNVFP4Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        int packedScales = segment.get(LE_INT, blockOffset);
        long quantsOffset = blockOffset + NVFP4_SUB_BLOCKS;
        return dotRowNVFP4SubBlock(segment, quantsOffset, vector, vectorBase, packedScales, 0)
                + dotRowNVFP4SubBlock(
                        segment,
                        quantsOffset + NVFP4_SUB_BLOCK_SIZE / 2L,
                        vector,
                        vectorBase + NVFP4_SUB_BLOCK_SIZE,
                        packedScales,
                        8)
                + dotRowNVFP4SubBlock(
                        segment,
                        quantsOffset + NVFP4_SUB_BLOCK_SIZE,
                        vector,
                        vectorBase + 2 * NVFP4_SUB_BLOCK_SIZE,
                        packedScales,
                        16)
                + dotRowNVFP4SubBlock(
                        segment,
                        quantsOffset + 3L * NVFP4_SUB_BLOCK_SIZE / 2L,
                        vector,
                        vectorBase + 3 * NVFP4_SUB_BLOCK_SIZE,
                        packedScales,
                        24);
    }

    static void dotRowsNVFP4_4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / NVFP4_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowNVFP4Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowNVFP4Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowNVFP4Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowNVFP4Block(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowNVFP4Block(
                    segment,
                    row0Offset + blockOffset + NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + NVFP4_BLOCK_SIZE);
            row1Sum1 += dotRowNVFP4Block(
                    segment,
                    row1Offset + blockOffset + NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + NVFP4_BLOCK_SIZE);
            row2Sum1 += dotRowNVFP4Block(
                    segment,
                    row2Offset + blockOffset + NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + NVFP4_BLOCK_SIZE);
            row3Sum1 += dotRowNVFP4Block(
                    segment,
                    row3Offset + blockOffset + NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + NVFP4_BLOCK_SIZE);
            row0Sum2 += dotRowNVFP4Block(
                    segment,
                    row0Offset + blockOffset + 2L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * NVFP4_BLOCK_SIZE);
            row1Sum2 += dotRowNVFP4Block(
                    segment,
                    row1Offset + blockOffset + 2L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * NVFP4_BLOCK_SIZE);
            row2Sum2 += dotRowNVFP4Block(
                    segment,
                    row2Offset + blockOffset + 2L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * NVFP4_BLOCK_SIZE);
            row3Sum2 += dotRowNVFP4Block(
                    segment,
                    row3Offset + blockOffset + 2L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * NVFP4_BLOCK_SIZE);
            row0Sum3 += dotRowNVFP4Block(
                    segment,
                    row0Offset + blockOffset + 3L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * NVFP4_BLOCK_SIZE);
            row1Sum3 += dotRowNVFP4Block(
                    segment,
                    row1Offset + blockOffset + 3L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * NVFP4_BLOCK_SIZE);
            row2Sum3 += dotRowNVFP4Block(
                    segment,
                    row2Offset + blockOffset + 3L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * NVFP4_BLOCK_SIZE);
            row3Sum3 += dotRowNVFP4Block(
                    segment,
                    row3Offset + blockOffset + 3L * NVFP4_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * NVFP4_BLOCK_SIZE);
            blockOffset += 4L * NVFP4_BLOCK_BYTES;
            vectorBase += 4 * NVFP4_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowNVFP4Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowNVFP4Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowNVFP4Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowNVFP4Block(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += NVFP4_BLOCK_BYTES;
            vectorBase += NVFP4_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowIQ4NL(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / IQ4_NL_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowIQ4NLBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowIQ4NLBlock(
                    segment,
                    blockOffset + IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + IQ4_NL_BLOCK_SIZE);
            sum2 += dotRowIQ4NLBlock(
                    segment,
                    blockOffset + 2L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * IQ4_NL_BLOCK_SIZE);
            sum3 += dotRowIQ4NLBlock(
                    segment,
                    blockOffset + 3L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * IQ4_NL_BLOCK_SIZE);
            blockOffset += 4L * IQ4_NL_BLOCK_BYTES;
            vectorBase += 4 * IQ4_NL_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowIQ4NLBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += IQ4_NL_BLOCK_BYTES;
            vectorBase += IQ4_NL_BLOCK_SIZE;
        }
        return sum;
    }

    private static float dotRowIQ4NLBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        long quantsOffset = blockOffset + 2;
        return d * dotRowNibble32(segment, quantsOffset, IQ4_NL_NIBBLE_PAIRS, vector, vectorBase);
    }

    static void dotRowsIQ4NL_4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / IQ4_NL_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowIQ4NLBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowIQ4NLBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowIQ4NLBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowIQ4NLBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowIQ4NLBlock(
                    segment,
                    row0Offset + blockOffset + IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + IQ4_NL_BLOCK_SIZE);
            row1Sum1 += dotRowIQ4NLBlock(
                    segment,
                    row1Offset + blockOffset + IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + IQ4_NL_BLOCK_SIZE);
            row2Sum1 += dotRowIQ4NLBlock(
                    segment,
                    row2Offset + blockOffset + IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + IQ4_NL_BLOCK_SIZE);
            row3Sum1 += dotRowIQ4NLBlock(
                    segment,
                    row3Offset + blockOffset + IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + IQ4_NL_BLOCK_SIZE);
            row0Sum2 += dotRowIQ4NLBlock(
                    segment,
                    row0Offset + blockOffset + 2L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * IQ4_NL_BLOCK_SIZE);
            row1Sum2 += dotRowIQ4NLBlock(
                    segment,
                    row1Offset + blockOffset + 2L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * IQ4_NL_BLOCK_SIZE);
            row2Sum2 += dotRowIQ4NLBlock(
                    segment,
                    row2Offset + blockOffset + 2L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * IQ4_NL_BLOCK_SIZE);
            row3Sum2 += dotRowIQ4NLBlock(
                    segment,
                    row3Offset + blockOffset + 2L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * IQ4_NL_BLOCK_SIZE);
            row0Sum3 += dotRowIQ4NLBlock(
                    segment,
                    row0Offset + blockOffset + 3L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * IQ4_NL_BLOCK_SIZE);
            row1Sum3 += dotRowIQ4NLBlock(
                    segment,
                    row1Offset + blockOffset + 3L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * IQ4_NL_BLOCK_SIZE);
            row2Sum3 += dotRowIQ4NLBlock(
                    segment,
                    row2Offset + blockOffset + 3L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * IQ4_NL_BLOCK_SIZE);
            row3Sum3 += dotRowIQ4NLBlock(
                    segment,
                    row3Offset + blockOffset + 3L * IQ4_NL_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * IQ4_NL_BLOCK_SIZE);
            blockOffset += 4L * IQ4_NL_BLOCK_BYTES;
            vectorBase += 4 * IQ4_NL_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowIQ4NLBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowIQ4NLBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowIQ4NLBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowIQ4NLBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += IQ4_NL_BLOCK_BYTES;
            vectorBase += IQ4_NL_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowIQ4XS(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowIQ4XSBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowIQ4XSBlock(
                    segment,
                    blockOffset + IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            sum2 += dotRowIQ4XSBlock(
                    segment,
                    blockOffset + 2L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            sum3 += dotRowIQ4XSBlock(
                    segment,
                    blockOffset + 3L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            blockOffset += 4L * IQ4_XS_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowIQ4XSBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += IQ4_XS_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    private static float dotRowIQ4XSBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        int scalesH = segment.get(LE_SHORT, blockOffset + 2) & 0xFFFF;
        long scalesLOffset = blockOffset + 4;
        int scalesL = segment.get(LE_INT, scalesLOffset);
        long quantsOffset = blockOffset + 8;
        return dotRowIQ4XSGroup(segment, quantsOffset, vector, vectorBase, d, scalesH, scalesL, 0)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 1L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 1)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 2L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + 2 * IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 2)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 3L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + 3 * IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 3)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 4L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + 4 * IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 4)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 5L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + 5 * IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 5)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 6L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + 6 * IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 6)
                + dotRowIQ4XSGroup(
                        segment, quantsOffset + 7L * IQ4_XS_GROUP_SIZE / 2L, vector,
                        vectorBase + 7 * IQ4_XS_GROUP_SIZE, d, scalesH, scalesL, 7);
    }

    static void dotRowsIQ4XS_4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int outputOffset) {
        float row0Sum0 = 0.0f;
        float row0Sum1 = 0.0f;
        float row0Sum2 = 0.0f;
        float row0Sum3 = 0.0f;
        float row1Sum0 = 0.0f;
        float row1Sum1 = 0.0f;
        float row1Sum2 = 0.0f;
        float row1Sum3 = 0.0f;
        float row2Sum0 = 0.0f;
        float row2Sum1 = 0.0f;
        float row2Sum2 = 0.0f;
        float row2Sum3 = 0.0f;
        float row3Sum0 = 0.0f;
        float row3Sum1 = 0.0f;
        float row3Sum2 = 0.0f;
        float row3Sum3 = 0.0f;
        long row0Offset = rowOffset;
        long row1Offset = rowOffset + rowBytes;
        long row2Offset = rowOffset + 2L * rowBytes;
        long row3Offset = rowOffset + 3L * rowBytes;
        long blockOffset = 0L;
        int vectorBase = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowIQ4XSBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowIQ4XSBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowIQ4XSBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowIQ4XSBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowIQ4XSBlock(
                    segment,
                    row0Offset + blockOffset + IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row1Sum1 += dotRowIQ4XSBlock(
                    segment,
                    row1Offset + blockOffset + IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row2Sum1 += dotRowIQ4XSBlock(
                    segment,
                    row2Offset + blockOffset + IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row3Sum1 += dotRowIQ4XSBlock(
                    segment,
                    row3Offset + blockOffset + IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            row0Sum2 += dotRowIQ4XSBlock(
                    segment,
                    row0Offset + blockOffset + 2L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row1Sum2 += dotRowIQ4XSBlock(
                    segment,
                    row1Offset + blockOffset + 2L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row2Sum2 += dotRowIQ4XSBlock(
                    segment,
                    row2Offset + blockOffset + 2L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row3Sum2 += dotRowIQ4XSBlock(
                    segment,
                    row3Offset + blockOffset + 2L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            row0Sum3 += dotRowIQ4XSBlock(
                    segment,
                    row0Offset + blockOffset + 3L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            row1Sum3 += dotRowIQ4XSBlock(
                    segment,
                    row1Offset + blockOffset + 3L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            row2Sum3 += dotRowIQ4XSBlock(
                    segment,
                    row2Offset + blockOffset + 3L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            row3Sum3 += dotRowIQ4XSBlock(
                    segment,
                    row3Offset + blockOffset + 3L * IQ4_XS_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            blockOffset += 4L * IQ4_XS_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowIQ4XSBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowIQ4XSBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowIQ4XSBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowIQ4XSBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += IQ4_XS_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowNibble16(
            MemorySegment segment,
            long quantsOffset,
            byte[] pairValues,
            float[] vector,
            int vectorOffset) {
        long packed = segment.get(LE_LONG, quantsOffset);
        return dotPackedNibbleByte8(packed, 0, pairValues, vector, vectorOffset, 0)
                + dotPackedNibbleByte8(packed, 8, pairValues, vector, vectorOffset, 1)
                + dotPackedNibbleByte8(packed, 16, pairValues, vector, vectorOffset, 2)
                + dotPackedNibbleByte8(packed, 24, pairValues, vector, vectorOffset, 3)
                + dotPackedNibbleByte8(packed, 32, pairValues, vector, vectorOffset, 4)
                + dotPackedNibbleByte8(packed, 40, pairValues, vector, vectorOffset, 5)
                + dotPackedNibbleByte8(packed, 48, pairValues, vector, vectorOffset, 6)
                + dotPackedNibbleByte8(packed, 56, pairValues, vector, vectorOffset, 7);
    }

    private static float dotRowNibble32(
            MemorySegment segment,
            long quantsOffset,
            byte[] pairValues,
            float[] vector,
            int vectorOffset) {
        long first = segment.get(LE_LONG, quantsOffset);
        long second = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        return dotPackedNibbleByte16(first, 0, pairValues, vector, vectorOffset, 0)
                + dotPackedNibbleByte16(first, 8, pairValues, vector, vectorOffset, 1)
                + dotPackedNibbleByte16(first, 16, pairValues, vector, vectorOffset, 2)
                + dotPackedNibbleByte16(first, 24, pairValues, vector, vectorOffset, 3)
                + dotPackedNibbleByte16(first, 32, pairValues, vector, vectorOffset, 4)
                + dotPackedNibbleByte16(first, 40, pairValues, vector, vectorOffset, 5)
                + dotPackedNibbleByte16(first, 48, pairValues, vector, vectorOffset, 6)
                + dotPackedNibbleByte16(first, 56, pairValues, vector, vectorOffset, 7)
                + dotPackedNibbleByte16(second, 0, pairValues, vector, vectorOffset, 8)
                + dotPackedNibbleByte16(second, 8, pairValues, vector, vectorOffset, 9)
                + dotPackedNibbleByte16(second, 16, pairValues, vector, vectorOffset, 10)
                + dotPackedNibbleByte16(second, 24, pairValues, vector, vectorOffset, 11)
                + dotPackedNibbleByte16(second, 32, pairValues, vector, vectorOffset, 12)
                + dotPackedNibbleByte16(second, 40, pairValues, vector, vectorOffset, 13)
                + dotPackedNibbleByte16(second, 48, pairValues, vector, vectorOffset, 14)
                + dotPackedNibbleByte16(second, 56, pairValues, vector, vectorOffset, 15);
    }

    private static float dotRowNVFP4SubBlock(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset,
            int packedScales,
            int scaleShift) {
        float d = ue4m3ToF32(unsignedByte(packedScales, scaleShift));
        return d * dotRowNibble16(segment, quantsOffset, MXFP4_NIBBLE_PAIRS, vector, vectorOffset);
    }

    private static float dotRowIQ4XSGroup(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset,
            float d,
            int scalesH,
            int scalesL,
            int group) {
        float dl = d * (iq4XSScalePacked(scalesH, scalesL, group) - 32);
        return dl * dotRowNibble32(segment, quantsOffset, IQ4_NL_NIBBLE_PAIRS, vector, vectorOffset);
    }

    private static float dotPackedNibbleByte8(
            long packed,
            int shift,
            byte[] pairValues,
            float[] vector,
            int vectorOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        int pairBase = nibblePairBase(quant);
        return pairValues[pairBase] * vector[vectorOffset + index]
                + pairValues[pairBase + 1] * vector[vectorOffset + 8 + index];
    }

    private static float dotPackedNibbleByte16(
            long packed,
            int shift,
            byte[] pairValues,
            float[] vector,
            int vectorOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        int pairBase = nibblePairBase(quant);
        return pairValues[pairBase] * vector[vectorOffset + index]
                + pairValues[pairBase + 1] * vector[vectorOffset + 16 + index];
    }
}
