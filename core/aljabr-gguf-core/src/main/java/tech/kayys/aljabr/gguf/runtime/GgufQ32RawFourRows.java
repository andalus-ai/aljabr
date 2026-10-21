package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.dotRowQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.dotRowQ4_1NoBiasBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.dotRowQ4_1PrecomputedBiasBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.dotRowQ5_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.dotRowQ5_1NoBiasBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32RawDot.dotRowQ5_1PrecomputedBiasBlock;

import java.lang.foreign.MemorySegment;

/**
 * Four-row raw Q32-family row-dot reducers.
 *
 * <p>Raw Q4/Q5 formats are still unpacked block-by-block, but mat-vec row
 * walkers commonly consume four adjacent rows at a time. Keeping that fused
 * traversal here avoids repeating the single-row setup and preserves the exact
 * per-row summation order from {@link GgufQ32RawDot}.</p>
 * @author bhangun
 */
final class GgufQ32RawFourRows {
    private GgufQ32RawFourRows() {
    }

    static void dotRowsQ4_0(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        long row0Block = rowOffset;
        long row1Block = rowOffset + rowBytes;
        long row2Block = rowOffset + 2L * rowBytes;
        long row3Block = rowOffset + 3L * rowBytes;
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
        int vectorBase = vectorOffset;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ4_0Block(segment, row0Block, vector, vectorBase);
            row1Sum0 += dotRowQ4_0Block(segment, row1Block, vector, vectorBase);
            row2Sum0 += dotRowQ4_0Block(segment, row2Block, vector, vectorBase);
            row3Sum0 += dotRowQ4_0Block(segment, row3Block, vector, vectorBase);

            int vector1 = vectorBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += dotRowQ4_0Block(segment, row0Block + Q4_0_BLOCK_BYTES, vector, vector1);
            row1Sum1 += dotRowQ4_0Block(segment, row1Block + Q4_0_BLOCK_BYTES, vector, vector1);
            row2Sum1 += dotRowQ4_0Block(segment, row2Block + Q4_0_BLOCK_BYTES, vector, vector1);
            row3Sum1 += dotRowQ4_0Block(segment, row3Block + Q4_0_BLOCK_BYTES, vector, vector1);

            int vector2 = vectorBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += dotRowQ4_0Block(segment, row0Block + 2L * Q4_0_BLOCK_BYTES, vector, vector2);
            row1Sum2 += dotRowQ4_0Block(segment, row1Block + 2L * Q4_0_BLOCK_BYTES, vector, vector2);
            row2Sum2 += dotRowQ4_0Block(segment, row2Block + 2L * Q4_0_BLOCK_BYTES, vector, vector2);
            row3Sum2 += dotRowQ4_0Block(segment, row3Block + 2L * Q4_0_BLOCK_BYTES, vector, vector2);

            int vector3 = vectorBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += dotRowQ4_0Block(segment, row0Block + 3L * Q4_0_BLOCK_BYTES, vector, vector3);
            row1Sum3 += dotRowQ4_0Block(segment, row1Block + 3L * Q4_0_BLOCK_BYTES, vector, vector3);
            row2Sum3 += dotRowQ4_0Block(segment, row2Block + 3L * Q4_0_BLOCK_BYTES, vector, vector3);
            row3Sum3 += dotRowQ4_0Block(segment, row3Block + 3L * Q4_0_BLOCK_BYTES, vector, vector3);

            row0Block += 4L * Q4_0_BLOCK_BYTES;
            row1Block += 4L * Q4_0_BLOCK_BYTES;
            row2Block += 4L * Q4_0_BLOCK_BYTES;
            row3Block += 4L * Q4_0_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ4_0Block(segment, row0Block, vector, vectorBase);
            row1Sum += dotRowQ4_0Block(segment, row1Block, vector, vectorBase);
            row2Sum += dotRowQ4_0Block(segment, row2Block, vector, vectorBase);
            row3Sum += dotRowQ4_0Block(segment, row3Block, vector, vectorBase);
            row0Block += Q4_0_BLOCK_BYTES;
            row1Block += Q4_0_BLOCK_BYTES;
            row2Block += Q4_0_BLOCK_BYTES;
            row3Block += Q4_0_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    static void dotRowsQ4_1NoBias(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        dotRowsQ4_1NoBiasBlocks(segment, rowOffset, rowBytes, columns, vector, vectorOffset, output, outputOffset);
    }

    static void dotRowsQ4_1(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] vectorBlockSums,
            float[] output,
            int outputOffset) {
        dotRowsQ4_1BiasBlocks(segment, rowOffset, rowBytes, columns, vector, vectorBlockSums, output, outputOffset);
    }

    static void dotRowsQ5_0(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        dotRowsQ5_0Blocks(segment, rowOffset, rowBytes, columns, vector, vectorOffset, output, outputOffset);
    }

    static void dotRowsQ5_1NoBias(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        dotRowsQ5_1NoBiasBlocks(segment, rowOffset, rowBytes, columns, vector, vectorOffset, output, outputOffset);
    }

    static void dotRowsQ5_1(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] vectorBlockSums,
            float[] output,
            int outputOffset) {
        dotRowsQ5_1BiasBlocks(segment, rowOffset, rowBytes, columns, vector, vectorBlockSums, output, outputOffset);
    }

    private static void dotRowsQ4_1NoBiasBlocks(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        long row0Block = rowOffset;
        long row1Block = rowOffset + rowBytes;
        long row2Block = rowOffset + 2L * rowBytes;
        long row3Block = rowOffset + 3L * rowBytes;
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
        int vectorBase = vectorOffset;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ4_1NoBiasBlock(segment, row0Block, vector, vectorBase);
            row1Sum0 += dotRowQ4_1NoBiasBlock(segment, row1Block, vector, vectorBase);
            row2Sum0 += dotRowQ4_1NoBiasBlock(segment, row2Block, vector, vectorBase);
            row3Sum0 += dotRowQ4_1NoBiasBlock(segment, row3Block, vector, vectorBase);
            int vector1 = vectorBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += dotRowQ4_1NoBiasBlock(segment, row0Block + Q4_1_BLOCK_BYTES, vector, vector1);
            row1Sum1 += dotRowQ4_1NoBiasBlock(segment, row1Block + Q4_1_BLOCK_BYTES, vector, vector1);
            row2Sum1 += dotRowQ4_1NoBiasBlock(segment, row2Block + Q4_1_BLOCK_BYTES, vector, vector1);
            row3Sum1 += dotRowQ4_1NoBiasBlock(segment, row3Block + Q4_1_BLOCK_BYTES, vector, vector1);
            int vector2 = vectorBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += dotRowQ4_1NoBiasBlock(segment, row0Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2);
            row1Sum2 += dotRowQ4_1NoBiasBlock(segment, row1Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2);
            row2Sum2 += dotRowQ4_1NoBiasBlock(segment, row2Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2);
            row3Sum2 += dotRowQ4_1NoBiasBlock(segment, row3Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2);
            int vector3 = vectorBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += dotRowQ4_1NoBiasBlock(segment, row0Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3);
            row1Sum3 += dotRowQ4_1NoBiasBlock(segment, row1Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3);
            row2Sum3 += dotRowQ4_1NoBiasBlock(segment, row2Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3);
            row3Sum3 += dotRowQ4_1NoBiasBlock(segment, row3Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3);
            row0Block += 4L * Q4_1_BLOCK_BYTES;
            row1Block += 4L * Q4_1_BLOCK_BYTES;
            row2Block += 4L * Q4_1_BLOCK_BYTES;
            row3Block += 4L * Q4_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ4_1NoBiasBlock(segment, row0Block, vector, vectorBase);
            row1Sum += dotRowQ4_1NoBiasBlock(segment, row1Block, vector, vectorBase);
            row2Sum += dotRowQ4_1NoBiasBlock(segment, row2Block, vector, vectorBase);
            row3Sum += dotRowQ4_1NoBiasBlock(segment, row3Block, vector, vectorBase);
            row0Block += Q4_1_BLOCK_BYTES;
            row1Block += Q4_1_BLOCK_BYTES;
            row2Block += Q4_1_BLOCK_BYTES;
            row3Block += Q4_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static void dotRowsQ4_1BiasBlocks(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] vectorBlockSums,
            float[] output,
            int outputOffset) {
        long row0Block = rowOffset;
        long row1Block = rowOffset + rowBytes;
        long row2Block = rowOffset + 2L * rowBytes;
        long row3Block = rowOffset + 3L * rowBytes;
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
        int vectorBase = 0;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ4_1PrecomputedBiasBlock(segment, row0Block, vector, vectorBase, vectorBlockSums[block]);
            row1Sum0 += dotRowQ4_1PrecomputedBiasBlock(segment, row1Block, vector, vectorBase, vectorBlockSums[block]);
            row2Sum0 += dotRowQ4_1PrecomputedBiasBlock(segment, row2Block, vector, vectorBase, vectorBlockSums[block]);
            row3Sum0 += dotRowQ4_1PrecomputedBiasBlock(segment, row3Block, vector, vectorBase, vectorBlockSums[block]);
            int vector1 = vectorBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row0Block + Q4_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            row1Sum1 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row1Block + Q4_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            row2Sum1 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row2Block + Q4_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            row3Sum1 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row3Block + Q4_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            int vector2 = vectorBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row0Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            row1Sum2 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row1Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            row2Sum2 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row2Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            row3Sum2 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row3Block + 2L * Q4_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            int vector3 = vectorBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row0Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row1Sum3 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row1Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row2Sum3 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row2Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row3Sum3 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, row3Block + 3L * Q4_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row0Block += 4L * Q4_1_BLOCK_BYTES;
            row1Block += 4L * Q4_1_BLOCK_BYTES;
            row2Block += 4L * Q4_1_BLOCK_BYTES;
            row3Block += 4L * Q4_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            float vectorBlockSum = vectorBlockSums[block];
            row0Sum += dotRowQ4_1PrecomputedBiasBlock(segment, row0Block, vector, vectorBase, vectorBlockSum);
            row1Sum += dotRowQ4_1PrecomputedBiasBlock(segment, row1Block, vector, vectorBase, vectorBlockSum);
            row2Sum += dotRowQ4_1PrecomputedBiasBlock(segment, row2Block, vector, vectorBase, vectorBlockSum);
            row3Sum += dotRowQ4_1PrecomputedBiasBlock(segment, row3Block, vector, vectorBase, vectorBlockSum);
            row0Block += Q4_1_BLOCK_BYTES;
            row1Block += Q4_1_BLOCK_BYTES;
            row2Block += Q4_1_BLOCK_BYTES;
            row3Block += Q4_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static void dotRowsQ5_0Blocks(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        long row0Block = rowOffset;
        long row1Block = rowOffset + rowBytes;
        long row2Block = rowOffset + 2L * rowBytes;
        long row3Block = rowOffset + 3L * rowBytes;
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
        int vectorBase = vectorOffset;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ5_0Block(segment, row0Block, vector, vectorBase);
            row1Sum0 += dotRowQ5_0Block(segment, row1Block, vector, vectorBase);
            row2Sum0 += dotRowQ5_0Block(segment, row2Block, vector, vectorBase);
            row3Sum0 += dotRowQ5_0Block(segment, row3Block, vector, vectorBase);
            int vector1 = vectorBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += dotRowQ5_0Block(segment, row0Block + Q5_0_BLOCK_BYTES, vector, vector1);
            row1Sum1 += dotRowQ5_0Block(segment, row1Block + Q5_0_BLOCK_BYTES, vector, vector1);
            row2Sum1 += dotRowQ5_0Block(segment, row2Block + Q5_0_BLOCK_BYTES, vector, vector1);
            row3Sum1 += dotRowQ5_0Block(segment, row3Block + Q5_0_BLOCK_BYTES, vector, vector1);
            int vector2 = vectorBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += dotRowQ5_0Block(segment, row0Block + 2L * Q5_0_BLOCK_BYTES, vector, vector2);
            row1Sum2 += dotRowQ5_0Block(segment, row1Block + 2L * Q5_0_BLOCK_BYTES, vector, vector2);
            row2Sum2 += dotRowQ5_0Block(segment, row2Block + 2L * Q5_0_BLOCK_BYTES, vector, vector2);
            row3Sum2 += dotRowQ5_0Block(segment, row3Block + 2L * Q5_0_BLOCK_BYTES, vector, vector2);
            int vector3 = vectorBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += dotRowQ5_0Block(segment, row0Block + 3L * Q5_0_BLOCK_BYTES, vector, vector3);
            row1Sum3 += dotRowQ5_0Block(segment, row1Block + 3L * Q5_0_BLOCK_BYTES, vector, vector3);
            row2Sum3 += dotRowQ5_0Block(segment, row2Block + 3L * Q5_0_BLOCK_BYTES, vector, vector3);
            row3Sum3 += dotRowQ5_0Block(segment, row3Block + 3L * Q5_0_BLOCK_BYTES, vector, vector3);
            row0Block += 4L * Q5_0_BLOCK_BYTES;
            row1Block += 4L * Q5_0_BLOCK_BYTES;
            row2Block += 4L * Q5_0_BLOCK_BYTES;
            row3Block += 4L * Q5_0_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ5_0Block(segment, row0Block, vector, vectorBase);
            row1Sum += dotRowQ5_0Block(segment, row1Block, vector, vectorBase);
            row2Sum += dotRowQ5_0Block(segment, row2Block, vector, vectorBase);
            row3Sum += dotRowQ5_0Block(segment, row3Block, vector, vectorBase);
            row0Block += Q5_0_BLOCK_BYTES;
            row1Block += Q5_0_BLOCK_BYTES;
            row2Block += Q5_0_BLOCK_BYTES;
            row3Block += Q5_0_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static void dotRowsQ5_1NoBiasBlocks(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] output,
            int outputOffset) {
        long row0Block = rowOffset;
        long row1Block = rowOffset + rowBytes;
        long row2Block = rowOffset + 2L * rowBytes;
        long row3Block = rowOffset + 3L * rowBytes;
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
        int vectorBase = vectorOffset;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ5_1NoBiasBlock(segment, row0Block, vector, vectorBase);
            row1Sum0 += dotRowQ5_1NoBiasBlock(segment, row1Block, vector, vectorBase);
            row2Sum0 += dotRowQ5_1NoBiasBlock(segment, row2Block, vector, vectorBase);
            row3Sum0 += dotRowQ5_1NoBiasBlock(segment, row3Block, vector, vectorBase);
            int vector1 = vectorBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += dotRowQ5_1NoBiasBlock(segment, row0Block + Q5_1_BLOCK_BYTES, vector, vector1);
            row1Sum1 += dotRowQ5_1NoBiasBlock(segment, row1Block + Q5_1_BLOCK_BYTES, vector, vector1);
            row2Sum1 += dotRowQ5_1NoBiasBlock(segment, row2Block + Q5_1_BLOCK_BYTES, vector, vector1);
            row3Sum1 += dotRowQ5_1NoBiasBlock(segment, row3Block + Q5_1_BLOCK_BYTES, vector, vector1);
            int vector2 = vectorBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += dotRowQ5_1NoBiasBlock(segment, row0Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2);
            row1Sum2 += dotRowQ5_1NoBiasBlock(segment, row1Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2);
            row2Sum2 += dotRowQ5_1NoBiasBlock(segment, row2Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2);
            row3Sum2 += dotRowQ5_1NoBiasBlock(segment, row3Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2);
            int vector3 = vectorBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += dotRowQ5_1NoBiasBlock(segment, row0Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3);
            row1Sum3 += dotRowQ5_1NoBiasBlock(segment, row1Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3);
            row2Sum3 += dotRowQ5_1NoBiasBlock(segment, row2Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3);
            row3Sum3 += dotRowQ5_1NoBiasBlock(segment, row3Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3);
            row0Block += 4L * Q5_1_BLOCK_BYTES;
            row1Block += 4L * Q5_1_BLOCK_BYTES;
            row2Block += 4L * Q5_1_BLOCK_BYTES;
            row3Block += 4L * Q5_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ5_1NoBiasBlock(segment, row0Block, vector, vectorBase);
            row1Sum += dotRowQ5_1NoBiasBlock(segment, row1Block, vector, vectorBase);
            row2Sum += dotRowQ5_1NoBiasBlock(segment, row2Block, vector, vectorBase);
            row3Sum += dotRowQ5_1NoBiasBlock(segment, row3Block, vector, vectorBase);
            row0Block += Q5_1_BLOCK_BYTES;
            row1Block += Q5_1_BLOCK_BYTES;
            row2Block += Q5_1_BLOCK_BYTES;
            row3Block += Q5_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static void dotRowsQ5_1BiasBlocks(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] vectorBlockSums,
            float[] output,
            int outputOffset) {
        long row0Block = rowOffset;
        long row1Block = rowOffset + rowBytes;
        long row2Block = rowOffset + 2L * rowBytes;
        long row3Block = rowOffset + 3L * rowBytes;
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
        int vectorBase = 0;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ5_1PrecomputedBiasBlock(segment, row0Block, vector, vectorBase, vectorBlockSums[block]);
            row1Sum0 += dotRowQ5_1PrecomputedBiasBlock(segment, row1Block, vector, vectorBase, vectorBlockSums[block]);
            row2Sum0 += dotRowQ5_1PrecomputedBiasBlock(segment, row2Block, vector, vectorBase, vectorBlockSums[block]);
            row3Sum0 += dotRowQ5_1PrecomputedBiasBlock(segment, row3Block, vector, vectorBase, vectorBlockSums[block]);
            int vector1 = vectorBase + Q4_0_BLOCK_SIZE;
            row0Sum1 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row0Block + Q5_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            row1Sum1 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row1Block + Q5_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            row2Sum1 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row2Block + Q5_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            row3Sum1 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row3Block + Q5_1_BLOCK_BYTES, vector, vector1, vectorBlockSums[block + 1]);
            int vector2 = vectorBase + 2 * Q4_0_BLOCK_SIZE;
            row0Sum2 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row0Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            row1Sum2 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row1Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            row2Sum2 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row2Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            row3Sum2 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row3Block + 2L * Q5_1_BLOCK_BYTES, vector, vector2, vectorBlockSums[block + 2]);
            int vector3 = vectorBase + 3 * Q4_0_BLOCK_SIZE;
            row0Sum3 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row0Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row1Sum3 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row1Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row2Sum3 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row2Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row3Sum3 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, row3Block + 3L * Q5_1_BLOCK_BYTES, vector, vector3, vectorBlockSums[block + 3]);
            row0Block += 4L * Q5_1_BLOCK_BYTES;
            row1Block += 4L * Q5_1_BLOCK_BYTES;
            row2Block += 4L * Q5_1_BLOCK_BYTES;
            row3Block += 4L * Q5_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            float vectorBlockSum = vectorBlockSums[block];
            row0Sum += dotRowQ5_1PrecomputedBiasBlock(segment, row0Block, vector, vectorBase, vectorBlockSum);
            row1Sum += dotRowQ5_1PrecomputedBiasBlock(segment, row1Block, vector, vectorBase, vectorBlockSum);
            row2Sum += dotRowQ5_1PrecomputedBiasBlock(segment, row2Block, vector, vectorBase, vectorBlockSum);
            row3Sum += dotRowQ5_1PrecomputedBiasBlock(segment, row3Block, vector, vectorBase, vectorBlockSum);
            row0Block += Q5_1_BLOCK_BYTES;
            row1Block += Q5_1_BLOCK_BYTES;
            row2Block += Q5_1_BLOCK_BYTES;
            row3Block += Q5_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }
}
