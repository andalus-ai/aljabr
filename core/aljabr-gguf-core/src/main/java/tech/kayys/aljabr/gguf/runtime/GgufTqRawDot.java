package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.Q1_0_SIGNS;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.TQ1_0_TRITS;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.TQ2_0_LANES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.q1_0SignBase;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.tq1_0TritBase;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.tq2_0LaneBase;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q1_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q1_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_QUANT_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_SCALE_OFFSET;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_QUANT_BYTES;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw row-dot kernels for Q1 and ternary GGUF formats.
 *
 * <p>Q1_0, TQ1_0, and TQ2_0 are compact low-bit encodings that expand through
 * small lookup tables while reading raw {@link MemorySegment} data.</p>
 * @author bhangun
 */
final class GgufTqRawDot {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufTqRawDot() {
    }

    static float dotRowQ1_0(
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
        int blocks = columns / Q1_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ1_0Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ1_0Block(
                    segment,
                    blockOffset + Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + Q1_0_BLOCK_SIZE);
            sum2 += dotRowQ1_0Block(
                    segment,
                    blockOffset + 2L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q1_0_BLOCK_SIZE);
            sum3 += dotRowQ1_0Block(
                    segment,
                    blockOffset + 3L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q1_0_BLOCK_SIZE);
            blockOffset += 4L * Q1_0_BLOCK_BYTES;
            vectorBase += 4 * Q1_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ1_0Block(segment, blockOffset, vector, vectorBase);
            blockOffset += Q1_0_BLOCK_BYTES;
            vectorBase += Q1_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static float dotRowQ1_0Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        return d * dotRowQ1UnscaledBlock(segment, blockOffset + 2, vector, vectorBase);
    }

    static void dotRowsQ1_0_4(
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
        int blocks = columns / Q1_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ1_0Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowQ1_0Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowQ1_0Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowQ1_0Block(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowQ1_0Block(
                    segment,
                    row0Offset + blockOffset + Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + Q1_0_BLOCK_SIZE);
            row1Sum1 += dotRowQ1_0Block(
                    segment,
                    row1Offset + blockOffset + Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + Q1_0_BLOCK_SIZE);
            row2Sum1 += dotRowQ1_0Block(
                    segment,
                    row2Offset + blockOffset + Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + Q1_0_BLOCK_SIZE);
            row3Sum1 += dotRowQ1_0Block(
                    segment,
                    row3Offset + blockOffset + Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + Q1_0_BLOCK_SIZE);
            row0Sum2 += dotRowQ1_0Block(
                    segment,
                    row0Offset + blockOffset + 2L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q1_0_BLOCK_SIZE);
            row1Sum2 += dotRowQ1_0Block(
                    segment,
                    row1Offset + blockOffset + 2L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q1_0_BLOCK_SIZE);
            row2Sum2 += dotRowQ1_0Block(
                    segment,
                    row2Offset + blockOffset + 2L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q1_0_BLOCK_SIZE);
            row3Sum2 += dotRowQ1_0Block(
                    segment,
                    row3Offset + blockOffset + 2L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q1_0_BLOCK_SIZE);
            row0Sum3 += dotRowQ1_0Block(
                    segment,
                    row0Offset + blockOffset + 3L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q1_0_BLOCK_SIZE);
            row1Sum3 += dotRowQ1_0Block(
                    segment,
                    row1Offset + blockOffset + 3L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q1_0_BLOCK_SIZE);
            row2Sum3 += dotRowQ1_0Block(
                    segment,
                    row2Offset + blockOffset + 3L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q1_0_BLOCK_SIZE);
            row3Sum3 += dotRowQ1_0Block(
                    segment,
                    row3Offset + blockOffset + 3L * Q1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q1_0_BLOCK_SIZE);
            blockOffset += 4L * Q1_0_BLOCK_BYTES;
            vectorBase += 4 * Q1_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ1_0Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowQ1_0Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowQ1_0Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowQ1_0Block(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += Q1_0_BLOCK_BYTES;
            vectorBase += Q1_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowTQ1_0(
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
        int blocks = columns / TQ1_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowTQ1_0Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowTQ1_0Block(
                    segment,
                    blockOffset + TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ1_0_BLOCK_SIZE);
            sum2 += dotRowTQ1_0Block(
                    segment,
                    blockOffset + 2L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ1_0_BLOCK_SIZE);
            sum3 += dotRowTQ1_0Block(
                    segment,
                    blockOffset + 3L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ1_0_BLOCK_SIZE);
            blockOffset += 4L * TQ1_0_BLOCK_BYTES;
            vectorBase += 4 * TQ1_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowTQ1_0Block(segment, blockOffset, vector, vectorBase);
            blockOffset += TQ1_0_BLOCK_BYTES;
            vectorBase += TQ1_0_BLOCK_SIZE;
        }
        return sum;
    }

    private static float dotRowTQ1_0Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + TQ1_0_SCALE_OFFSET));
        float quantDot = dotRowTQ1_0Group32(segment, blockOffset, vector, vectorBase)
                + dotRowTQ1_0Group16(segment, blockOffset + 32, vector, vectorBase + 160);
        long highOffset = blockOffset + TQ1_0_QUANT_BYTES;
        quantDot += dotRowTQ1_0High(segment.get(LE_INT, highOffset), vector, vectorBase + 240);
        return d * quantDot;
    }

    static void dotRowsTQ1_0_4(
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
        int blocks = columns / TQ1_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowTQ1_0Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowTQ1_0Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowTQ1_0Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowTQ1_0Block(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowTQ1_0Block(
                    segment,
                    row0Offset + blockOffset + TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ1_0_BLOCK_SIZE);
            row1Sum1 += dotRowTQ1_0Block(
                    segment,
                    row1Offset + blockOffset + TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ1_0_BLOCK_SIZE);
            row2Sum1 += dotRowTQ1_0Block(
                    segment,
                    row2Offset + blockOffset + TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ1_0_BLOCK_SIZE);
            row3Sum1 += dotRowTQ1_0Block(
                    segment,
                    row3Offset + blockOffset + TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ1_0_BLOCK_SIZE);
            row0Sum2 += dotRowTQ1_0Block(
                    segment,
                    row0Offset + blockOffset + 2L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ1_0_BLOCK_SIZE);
            row1Sum2 += dotRowTQ1_0Block(
                    segment,
                    row1Offset + blockOffset + 2L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ1_0_BLOCK_SIZE);
            row2Sum2 += dotRowTQ1_0Block(
                    segment,
                    row2Offset + blockOffset + 2L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ1_0_BLOCK_SIZE);
            row3Sum2 += dotRowTQ1_0Block(
                    segment,
                    row3Offset + blockOffset + 2L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ1_0_BLOCK_SIZE);
            row0Sum3 += dotRowTQ1_0Block(
                    segment,
                    row0Offset + blockOffset + 3L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ1_0_BLOCK_SIZE);
            row1Sum3 += dotRowTQ1_0Block(
                    segment,
                    row1Offset + blockOffset + 3L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ1_0_BLOCK_SIZE);
            row2Sum3 += dotRowTQ1_0Block(
                    segment,
                    row2Offset + blockOffset + 3L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ1_0_BLOCK_SIZE);
            row3Sum3 += dotRowTQ1_0Block(
                    segment,
                    row3Offset + blockOffset + 3L * TQ1_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ1_0_BLOCK_SIZE);
            blockOffset += 4L * TQ1_0_BLOCK_BYTES;
            vectorBase += 4 * TQ1_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowTQ1_0Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowTQ1_0Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowTQ1_0Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowTQ1_0Block(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += TQ1_0_BLOCK_BYTES;
            vectorBase += TQ1_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static float dotRowTQ2_0(
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
        int blocks = columns / TQ2_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowTQ2_0Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowTQ2_0Block(
                    segment,
                    blockOffset + TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ2_0_BLOCK_SIZE);
            sum2 += dotRowTQ2_0Block(
                    segment,
                    blockOffset + 2L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ2_0_BLOCK_SIZE);
            sum3 += dotRowTQ2_0Block(
                    segment,
                    blockOffset + 3L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ2_0_BLOCK_SIZE);
            blockOffset += 4L * TQ2_0_BLOCK_BYTES;
            vectorBase += 4 * TQ2_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowTQ2_0Block(segment, blockOffset, vector, vectorBase);
            blockOffset += TQ2_0_BLOCK_BYTES;
            vectorBase += TQ2_0_BLOCK_SIZE;
        }
        return sum;
    }

    static void dotRowsTQ2_0_4(
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
        int blocks = columns / TQ2_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowTQ2_0Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowTQ2_0Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowTQ2_0Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowTQ2_0Block(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowTQ2_0Block(
                    segment,
                    row0Offset + blockOffset + TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ2_0_BLOCK_SIZE);
            row1Sum1 += dotRowTQ2_0Block(
                    segment,
                    row1Offset + blockOffset + TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ2_0_BLOCK_SIZE);
            row2Sum1 += dotRowTQ2_0Block(
                    segment,
                    row2Offset + blockOffset + TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ2_0_BLOCK_SIZE);
            row3Sum1 += dotRowTQ2_0Block(
                    segment,
                    row3Offset + blockOffset + TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + TQ2_0_BLOCK_SIZE);
            row0Sum2 += dotRowTQ2_0Block(
                    segment,
                    row0Offset + blockOffset + 2L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ2_0_BLOCK_SIZE);
            row1Sum2 += dotRowTQ2_0Block(
                    segment,
                    row1Offset + blockOffset + 2L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ2_0_BLOCK_SIZE);
            row2Sum2 += dotRowTQ2_0Block(
                    segment,
                    row2Offset + blockOffset + 2L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ2_0_BLOCK_SIZE);
            row3Sum2 += dotRowTQ2_0Block(
                    segment,
                    row3Offset + blockOffset + 2L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * TQ2_0_BLOCK_SIZE);
            row0Sum3 += dotRowTQ2_0Block(
                    segment,
                    row0Offset + blockOffset + 3L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ2_0_BLOCK_SIZE);
            row1Sum3 += dotRowTQ2_0Block(
                    segment,
                    row1Offset + blockOffset + 3L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ2_0_BLOCK_SIZE);
            row2Sum3 += dotRowTQ2_0Block(
                    segment,
                    row2Offset + blockOffset + 3L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ2_0_BLOCK_SIZE);
            row3Sum3 += dotRowTQ2_0Block(
                    segment,
                    row3Offset + blockOffset + 3L * TQ2_0_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * TQ2_0_BLOCK_SIZE);
            blockOffset += 4L * TQ2_0_BLOCK_BYTES;
            vectorBase += 4 * TQ2_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowTQ2_0Block(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowTQ2_0Block(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowTQ2_0Block(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowTQ2_0Block(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += TQ2_0_BLOCK_BYTES;
            vectorBase += TQ2_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowTQ2_0Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + TQ2_0_QUANT_BYTES));
        float quantDot = dotRowTQ2_0Group(segment, blockOffset, vector, vectorBase)
                + dotRowTQ2_0Group(segment, blockOffset + 32, vector, vectorBase + 128);
        return d * quantDot;
    }

    private static float dotRowQ1UnscaledBlock(
            MemorySegment segment,
            long bitsOffset,
            float[] vector,
            int vectorOffset) {
        long firstMasks = segment.get(LE_LONG, bitsOffset);
        long secondMasks = segment.get(LE_LONG, bitsOffset + Long.BYTES);
        float sum0 = dotPackedQ1Byte(firstMasks, 0, vector, vectorOffset)
                + dotPackedQ1Byte(firstMasks, 32, vector, vectorOffset + 32)
                + dotPackedQ1Byte(secondMasks, 0, vector, vectorOffset + 64)
                + dotPackedQ1Byte(secondMasks, 32, vector, vectorOffset + 96);
        float sum1 = dotPackedQ1Byte(firstMasks, 8, vector, vectorOffset + 8)
                + dotPackedQ1Byte(firstMasks, 40, vector, vectorOffset + 40)
                + dotPackedQ1Byte(secondMasks, 8, vector, vectorOffset + 72)
                + dotPackedQ1Byte(secondMasks, 40, vector, vectorOffset + 104);
        float sum2 = dotPackedQ1Byte(firstMasks, 16, vector, vectorOffset + 16)
                + dotPackedQ1Byte(firstMasks, 48, vector, vectorOffset + 48)
                + dotPackedQ1Byte(secondMasks, 16, vector, vectorOffset + 80)
                + dotPackedQ1Byte(secondMasks, 48, vector, vectorOffset + 112);
        float sum3 = dotPackedQ1Byte(firstMasks, 24, vector, vectorOffset + 24)
                + dotPackedQ1Byte(firstMasks, 56, vector, vectorOffset + 56)
                + dotPackedQ1Byte(secondMasks, 24, vector, vectorOffset + 88)
                + dotPackedQ1Byte(secondMasks, 56, vector, vectorOffset + 120);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotPackedQ1Byte(long masks, int shift, float[] vector, int vectorOffset) {
        int mask = unsignedByte(masks, shift);
        int signBase = q1_0SignBase(mask);
        return Q1_0_SIGNS[signBase] * vector[vectorOffset]
                + Q1_0_SIGNS[signBase + 1] * vector[vectorOffset + 1]
                + Q1_0_SIGNS[signBase + 2] * vector[vectorOffset + 2]
                + Q1_0_SIGNS[signBase + 3] * vector[vectorOffset + 3]
                + Q1_0_SIGNS[signBase + 4] * vector[vectorOffset + 4]
                + Q1_0_SIGNS[signBase + 5] * vector[vectorOffset + 5]
                + Q1_0_SIGNS[signBase + 6] * vector[vectorOffset + 6]
                + Q1_0_SIGNS[signBase + 7] * vector[vectorOffset + 7];
    }

    private static float dotRowTQ2_0Group(
            MemorySegment segment,
            long groupOffset,
            float[] vector,
            int groupBase) {
        long packed0 = segment.get(LE_LONG, groupOffset);
        long packed1 = segment.get(LE_LONG, groupOffset + Long.BYTES);
        long packed2 = segment.get(LE_LONG, groupOffset + 2L * Long.BYTES);
        long packed3 = segment.get(LE_LONG, groupOffset + 3L * Long.BYTES);
        float sum0 = dotPackedTQ2_0Byte(packed0, 0, vector, groupBase, 0)
                + dotPackedTQ2_0Byte(packed0, 32, vector, groupBase, 4)
                + dotPackedTQ2_0Byte(packed1, 0, vector, groupBase, 8)
                + dotPackedTQ2_0Byte(packed1, 32, vector, groupBase, 12)
                + dotPackedTQ2_0Byte(packed2, 0, vector, groupBase, 16)
                + dotPackedTQ2_0Byte(packed2, 32, vector, groupBase, 20)
                + dotPackedTQ2_0Byte(packed3, 0, vector, groupBase, 24)
                + dotPackedTQ2_0Byte(packed3, 32, vector, groupBase, 28);
        float sum1 = dotPackedTQ2_0Byte(packed0, 8, vector, groupBase, 1)
                + dotPackedTQ2_0Byte(packed0, 40, vector, groupBase, 5)
                + dotPackedTQ2_0Byte(packed1, 8, vector, groupBase, 9)
                + dotPackedTQ2_0Byte(packed1, 40, vector, groupBase, 13)
                + dotPackedTQ2_0Byte(packed2, 8, vector, groupBase, 17)
                + dotPackedTQ2_0Byte(packed2, 40, vector, groupBase, 21)
                + dotPackedTQ2_0Byte(packed3, 8, vector, groupBase, 25)
                + dotPackedTQ2_0Byte(packed3, 40, vector, groupBase, 29);
        float sum2 = dotPackedTQ2_0Byte(packed0, 16, vector, groupBase, 2)
                + dotPackedTQ2_0Byte(packed0, 48, vector, groupBase, 6)
                + dotPackedTQ2_0Byte(packed1, 16, vector, groupBase, 10)
                + dotPackedTQ2_0Byte(packed1, 48, vector, groupBase, 14)
                + dotPackedTQ2_0Byte(packed2, 16, vector, groupBase, 18)
                + dotPackedTQ2_0Byte(packed2, 48, vector, groupBase, 22)
                + dotPackedTQ2_0Byte(packed3, 16, vector, groupBase, 26)
                + dotPackedTQ2_0Byte(packed3, 48, vector, groupBase, 30);
        float sum3 = dotPackedTQ2_0Byte(packed0, 24, vector, groupBase, 3)
                + dotPackedTQ2_0Byte(packed0, 56, vector, groupBase, 7)
                + dotPackedTQ2_0Byte(packed1, 24, vector, groupBase, 11)
                + dotPackedTQ2_0Byte(packed1, 56, vector, groupBase, 15)
                + dotPackedTQ2_0Byte(packed2, 24, vector, groupBase, 19)
                + dotPackedTQ2_0Byte(packed2, 56, vector, groupBase, 23)
                + dotPackedTQ2_0Byte(packed3, 24, vector, groupBase, 27)
                + dotPackedTQ2_0Byte(packed3, 56, vector, groupBase, 31);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotPackedTQ2_0Byte(
            long packed,
            int shift,
            float[] vector,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int laneBase = tq2_0LaneBase(quant);
        return TQ2_0_LANES[laneBase] * vector[groupBase + index]
                + TQ2_0_LANES[laneBase + 1] * vector[groupBase + 32 + index]
                + TQ2_0_LANES[laneBase + 2] * vector[groupBase + 64 + index]
                + TQ2_0_LANES[laneBase + 3] * vector[groupBase + 96 + index];
    }

    private static float dotRowTQ1_0Group32(
            MemorySegment segment,
            long groupOffset,
            float[] vector,
            int groupBase) {
        float sum = 0.0f;
        sum += dotRowTQ1_0Packed32(segment.get(LE_LONG, groupOffset), vector, groupBase, 0);
        sum += dotRowTQ1_0Packed32(segment.get(LE_LONG, groupOffset + Long.BYTES), vector, groupBase, 8);
        sum += dotRowTQ1_0Packed32(segment.get(LE_LONG, groupOffset + 2L * Long.BYTES), vector, groupBase, 16);
        sum += dotRowTQ1_0Packed32(segment.get(LE_LONG, groupOffset + 3L * Long.BYTES), vector, groupBase, 24);
        return sum;
    }

    private static float dotRowTQ1_0Group16(
            MemorySegment segment,
            long groupOffset,
            float[] vector,
            int groupBase) {
        float sum = 0.0f;
        sum += dotRowTQ1_0Packed16(segment.get(LE_LONG, groupOffset), vector, groupBase, 0);
        sum += dotRowTQ1_0Packed16(segment.get(LE_LONG, groupOffset + Long.BYTES), vector, groupBase, 8);
        return sum;
    }

    private static float dotRowTQ1_0Packed32(long packed, float[] vector, int groupBase, int indexBase) {
        float sum = 0.0f;
        sum += dotPackedTQ1_0Byte32(packed, 0, vector, groupBase, indexBase);
        sum += dotPackedTQ1_0Byte32(packed, 8, vector, groupBase, indexBase + 1);
        sum += dotPackedTQ1_0Byte32(packed, 16, vector, groupBase, indexBase + 2);
        sum += dotPackedTQ1_0Byte32(packed, 24, vector, groupBase, indexBase + 3);
        sum += dotPackedTQ1_0Byte32(packed, 32, vector, groupBase, indexBase + 4);
        sum += dotPackedTQ1_0Byte32(packed, 40, vector, groupBase, indexBase + 5);
        sum += dotPackedTQ1_0Byte32(packed, 48, vector, groupBase, indexBase + 6);
        sum += dotPackedTQ1_0Byte32(packed, 56, vector, groupBase, indexBase + 7);
        return sum;
    }

    private static float dotRowTQ1_0Packed16(long packed, float[] vector, int groupBase, int indexBase) {
        float sum = 0.0f;
        sum += dotPackedTQ1_0Byte16(packed, 0, vector, groupBase, indexBase);
        sum += dotPackedTQ1_0Byte16(packed, 8, vector, groupBase, indexBase + 1);
        sum += dotPackedTQ1_0Byte16(packed, 16, vector, groupBase, indexBase + 2);
        sum += dotPackedTQ1_0Byte16(packed, 24, vector, groupBase, indexBase + 3);
        sum += dotPackedTQ1_0Byte16(packed, 32, vector, groupBase, indexBase + 4);
        sum += dotPackedTQ1_0Byte16(packed, 40, vector, groupBase, indexBase + 5);
        sum += dotPackedTQ1_0Byte16(packed, 48, vector, groupBase, indexBase + 6);
        sum += dotPackedTQ1_0Byte16(packed, 56, vector, groupBase, indexBase + 7);
        return sum;
    }

    private static float dotPackedTQ1_0Byte32(
            long packed,
            int shift,
            float[] vector,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        return TQ1_0_TRITS[tritBase] * vector[groupBase + index]
                + TQ1_0_TRITS[tritBase + 1] * vector[groupBase + 32 + index]
                + TQ1_0_TRITS[tritBase + 2] * vector[groupBase + 64 + index]
                + TQ1_0_TRITS[tritBase + 3] * vector[groupBase + 96 + index]
                + TQ1_0_TRITS[tritBase + 4] * vector[groupBase + 128 + index];
    }

    private static float dotPackedTQ1_0Byte16(
            long packed,
            int shift,
            float[] vector,
            int groupBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        return TQ1_0_TRITS[tritBase] * vector[groupBase + index]
                + TQ1_0_TRITS[tritBase + 1] * vector[groupBase + 16 + index]
                + TQ1_0_TRITS[tritBase + 2] * vector[groupBase + 32 + index]
                + TQ1_0_TRITS[tritBase + 3] * vector[groupBase + 48 + index]
                + TQ1_0_TRITS[tritBase + 4] * vector[groupBase + 64 + index];
    }

    private static float dotRowTQ1_0High(int highPacked, float[] vector, int highBase) {
        return dotPackedTQ1_0HighByte(highPacked, 0, vector, highBase, 0)
                + dotPackedTQ1_0HighByte(highPacked, 8, vector, highBase, 1)
                + dotPackedTQ1_0HighByte(highPacked, 16, vector, highBase, 2)
                + dotPackedTQ1_0HighByte(highPacked, 24, vector, highBase, 3);
    }

    private static float dotPackedTQ1_0HighByte(
            int packed,
            int shift,
            float[] vector,
            int highBase,
            int index) {
        int quant = unsignedByte(packed, shift);
        int tritBase = tq1_0TritBase(quant);
        return TQ1_0_TRITS[tritBase] * vector[highBase + index]
                + TQ1_0_TRITS[tritBase + 1] * vector[highBase + 4 + index]
                + TQ1_0_TRITS[tritBase + 2] * vector[highBase + 8 + index]
                + TQ1_0_TRITS[tritBase + 3] * vector[highBase + 12 + index];
    }
}
