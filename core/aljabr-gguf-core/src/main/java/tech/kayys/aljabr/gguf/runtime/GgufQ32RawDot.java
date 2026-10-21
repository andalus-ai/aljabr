package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw Q4/Q5 GGUF row-dot kernels for Q32-family formats.
 *
 * <p>Q4_0, Q4_1, Q5_0, and Q5_1 all consume 32-value blocks directly from
 * {@link MemorySegment} data. This helper owns their unpack-on-read arithmetic
 * so {@link GgufTensorOps} can stay focused on dispatch and scheduling.</p>
 * @author bhangun
 */
final class GgufQ32RawDot {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ32RawDot() {
    }

    static float dotRowQ4_0(
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
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ4_0Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ4_0Block(
                    segment, blockOffset + Q4_0_BLOCK_BYTES, vector, vectorBase + Q4_0_BLOCK_SIZE);
            sum2 += dotRowQ4_0Block(
                    segment, blockOffset + 2L * Q4_0_BLOCK_BYTES, vector, vectorBase + 2 * Q4_0_BLOCK_SIZE);
            sum3 += dotRowQ4_0Block(
                    segment, blockOffset + 3L * Q4_0_BLOCK_BYTES, vector, vectorBase + 3 * Q4_0_BLOCK_SIZE);
            blockOffset += 4L * Q4_0_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ4_0Block(segment, blockOffset, vector, vectorBase);
            blockOffset += Q4_0_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ4_1(
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
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ4_1BiasedBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ4_1BiasedBlock(
                    segment, blockOffset + Q4_1_BLOCK_BYTES, vector, vectorBase + Q4_0_BLOCK_SIZE);
            sum2 += dotRowQ4_1BiasedBlock(
                    segment, blockOffset + 2L * Q4_1_BLOCK_BYTES, vector, vectorBase + 2 * Q4_0_BLOCK_SIZE);
            sum3 += dotRowQ4_1BiasedBlock(
                    segment, blockOffset + 3L * Q4_1_BLOCK_BYTES, vector, vectorBase + 3 * Q4_0_BLOCK_SIZE);
            blockOffset += 4L * Q4_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ4_1BiasedBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q4_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ4_1NoBias(
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
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ4_1NoBiasBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ4_1NoBiasBlock(
                    segment, blockOffset + Q4_1_BLOCK_BYTES, vector, vectorBase + Q4_0_BLOCK_SIZE);
            sum2 += dotRowQ4_1NoBiasBlock(
                    segment, blockOffset + 2L * Q4_1_BLOCK_BYTES, vector, vectorBase + 2 * Q4_0_BLOCK_SIZE);
            sum3 += dotRowQ4_1NoBiasBlock(
                    segment, blockOffset + 3L * Q4_1_BLOCK_BYTES, vector, vectorBase + 3 * Q4_0_BLOCK_SIZE);
            blockOffset += 4L * Q4_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ4_1NoBiasBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q4_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ4_1(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            float[] vectorBlockSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = 0;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ4_1PrecomputedBiasBlock(
                    segment, blockOffset, vector, vectorBase, vectorBlockSums[block]);
            sum1 += dotRowQ4_1PrecomputedBiasBlock(
                    segment,
                    blockOffset + Q4_1_BLOCK_BYTES,
                    vector,
                    vectorBase + Q4_0_BLOCK_SIZE,
                    vectorBlockSums[block + 1]);
            sum2 += dotRowQ4_1PrecomputedBiasBlock(
                    segment,
                    blockOffset + 2L * Q4_1_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q4_0_BLOCK_SIZE,
                    vectorBlockSums[block + 2]);
            sum3 += dotRowQ4_1PrecomputedBiasBlock(
                    segment,
                    blockOffset + 3L * Q4_1_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q4_0_BLOCK_SIZE,
                    vectorBlockSums[block + 3]);
            blockOffset += 4L * Q4_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ4_1PrecomputedBiasBlock(
                    segment, blockOffset, vector, vectorBase, vectorBlockSums[block]);
            blockOffset += Q4_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ5_0(
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
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ5_0Block(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ5_0Block(
                    segment, blockOffset + Q5_0_BLOCK_BYTES, vector, vectorBase + Q4_0_BLOCK_SIZE);
            sum2 += dotRowQ5_0Block(
                    segment, blockOffset + 2L * Q5_0_BLOCK_BYTES, vector, vectorBase + 2 * Q4_0_BLOCK_SIZE);
            sum3 += dotRowQ5_0Block(
                    segment, blockOffset + 3L * Q5_0_BLOCK_BYTES, vector, vectorBase + 3 * Q4_0_BLOCK_SIZE);
            blockOffset += 4L * Q5_0_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ5_0Block(segment, blockOffset, vector, vectorBase);
            blockOffset += Q5_0_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ5_1NoBias(
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
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ5_1NoBiasBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ5_1NoBiasBlock(
                    segment, blockOffset + Q5_1_BLOCK_BYTES, vector, vectorBase + Q4_0_BLOCK_SIZE);
            sum2 += dotRowQ5_1NoBiasBlock(
                    segment, blockOffset + 2L * Q5_1_BLOCK_BYTES, vector, vectorBase + 2 * Q4_0_BLOCK_SIZE);
            sum3 += dotRowQ5_1NoBiasBlock(
                    segment, blockOffset + 3L * Q5_1_BLOCK_BYTES, vector, vectorBase + 3 * Q4_0_BLOCK_SIZE);
            blockOffset += 4L * Q5_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ5_1NoBiasBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q5_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ5_1(
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
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ5_1BiasedBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ5_1BiasedBlock(
                    segment, blockOffset + Q5_1_BLOCK_BYTES, vector, vectorBase + Q4_0_BLOCK_SIZE);
            sum2 += dotRowQ5_1BiasedBlock(
                    segment, blockOffset + 2L * Q5_1_BLOCK_BYTES, vector, vectorBase + 2 * Q4_0_BLOCK_SIZE);
            sum3 += dotRowQ5_1BiasedBlock(
                    segment, blockOffset + 3L * Q5_1_BLOCK_BYTES, vector, vectorBase + 3 * Q4_0_BLOCK_SIZE);
            blockOffset += 4L * Q5_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ5_1BiasedBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q5_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ5_1(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            float[] vectorBlockSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = 0;
        int blocks = columns / Q4_0_BLOCK_SIZE;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ5_1PrecomputedBiasBlock(
                    segment, blockOffset, vector, vectorBase, vectorBlockSums[block]);
            sum1 += dotRowQ5_1PrecomputedBiasBlock(
                    segment,
                    blockOffset + Q5_1_BLOCK_BYTES,
                    vector,
                    vectorBase + Q4_0_BLOCK_SIZE,
                    vectorBlockSums[block + 1]);
            sum2 += dotRowQ5_1PrecomputedBiasBlock(
                    segment,
                    blockOffset + 2L * Q5_1_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * Q4_0_BLOCK_SIZE,
                    vectorBlockSums[block + 2]);
            sum3 += dotRowQ5_1PrecomputedBiasBlock(
                    segment,
                    blockOffset + 3L * Q5_1_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * Q4_0_BLOCK_SIZE,
                    vectorBlockSums[block + 3]);
            blockOffset += 4L * Q5_1_BLOCK_BYTES;
            vectorBase += 4 * Q4_0_BLOCK_SIZE;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ5_1PrecomputedBiasBlock(
                    segment, blockOffset, vector, vectorBase, vectorBlockSums[block]);
            blockOffset += Q5_1_BLOCK_BYTES;
            vectorBase += Q4_0_BLOCK_SIZE;
        }
        return sum;
    }

    static float dotRowQ4_0Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        return d * dotRowQ4_0UnscaledBlock(segment, blockOffset + 2, vector, vectorBase);
    }

    static float dotRowQ4_1NoBiasBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        return d * dotRowQ4_1UnscaledBlock(segment, blockOffset + 4, vector, vectorBase);
    }

    private static float dotRowQ4_1BiasedBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        float quantDot = dotRowQ4_1UnscaledBlock(segment, blockOffset + 4, vector, vectorBase);
        return d * quantDot + m * GgufSum.sumFloat32(vector, vectorBase);
    }

    static float dotRowQ4_1PrecomputedBiasBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase,
            float vectorBlockSum) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        float quantDot = dotRowQ4_1UnscaledBlock(segment, blockOffset + 4, vector, vectorBase);
        return d * quantDot + m * vectorBlockSum;
    }

    static float dotRowQ5_0Block(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        int highBits = segment.get(LE_INT, blockOffset + 2);
        return d * dotRowQ5_0UnscaledBlock(segment, blockOffset + 6, highBits, vector, vectorBase);
    }

    static float dotRowQ5_1NoBiasBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        int highBits = segment.get(LE_INT, blockOffset + 4);
        return d * dotRowQ5_1UnscaledBlock(segment, blockOffset + 8, highBits, vector, vectorBase);
    }

    private static float dotRowQ5_1BiasedBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        int highBits = segment.get(LE_INT, blockOffset + 4);
        float quantDot = dotRowQ5_1UnscaledBlock(segment, blockOffset + 8, highBits, vector, vectorBase);
        return d * quantDot + m * GgufSum.sumFloat32(vector, vectorBase);
    }

    static float dotRowQ5_1PrecomputedBiasBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase,
            float vectorBlockSum) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset));
        float m = f16ToF32(segment.get(LE_SHORT, blockOffset + 2));
        int highBits = segment.get(LE_INT, blockOffset + 4);
        float quantDot = dotRowQ5_1UnscaledBlock(segment, blockOffset + 8, highBits, vector, vectorBase);
        return d * quantDot + m * vectorBlockSum;
    }

    private static float dotRowQ4_0UnscaledBlock(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset) {
        long first = segment.get(LE_LONG, quantsOffset);
        long second = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        float sum0 = dotQ4_0PackedNibbleByte(first, 0, vector, vectorOffset, 0)
                + dotQ4_0PackedNibbleByte(first, 32, vector, vectorOffset, 4)
                + dotQ4_0PackedNibbleByte(second, 0, vector, vectorOffset, 8)
                + dotQ4_0PackedNibbleByte(second, 32, vector, vectorOffset, 12);
        float sum1 = dotQ4_0PackedNibbleByte(first, 8, vector, vectorOffset, 1)
                + dotQ4_0PackedNibbleByte(first, 40, vector, vectorOffset, 5)
                + dotQ4_0PackedNibbleByte(second, 8, vector, vectorOffset, 9)
                + dotQ4_0PackedNibbleByte(second, 40, vector, vectorOffset, 13);
        float sum2 = dotQ4_0PackedNibbleByte(first, 16, vector, vectorOffset, 2)
                + dotQ4_0PackedNibbleByte(first, 48, vector, vectorOffset, 6)
                + dotQ4_0PackedNibbleByte(second, 16, vector, vectorOffset, 10)
                + dotQ4_0PackedNibbleByte(second, 48, vector, vectorOffset, 14);
        float sum3 = dotQ4_0PackedNibbleByte(first, 24, vector, vectorOffset, 3)
                + dotQ4_0PackedNibbleByte(first, 56, vector, vectorOffset, 7)
                + dotQ4_0PackedNibbleByte(second, 24, vector, vectorOffset, 11)
                + dotQ4_0PackedNibbleByte(second, 56, vector, vectorOffset, 15);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotQ4_0PackedNibbleByte(
            long packed,
            int shift,
            float[] vector,
            int vectorOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        return ((quant & 0x0F) - 8) * vector[vectorOffset + index]
                + ((quant >>> 4) - 8) * vector[vectorOffset + 16 + index];
    }

    private static float dotRowQ4_1UnscaledBlock(
            MemorySegment segment,
            long quantsOffset,
            float[] vector,
            int vectorOffset) {
        long first = segment.get(LE_LONG, quantsOffset);
        long second = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        float sum0 = dotQ4_1PackedNibbleByte(first, 0, vector, vectorOffset, 0)
                + dotQ4_1PackedNibbleByte(first, 32, vector, vectorOffset, 4)
                + dotQ4_1PackedNibbleByte(second, 0, vector, vectorOffset, 8)
                + dotQ4_1PackedNibbleByte(second, 32, vector, vectorOffset, 12);
        float sum1 = dotQ4_1PackedNibbleByte(first, 8, vector, vectorOffset, 1)
                + dotQ4_1PackedNibbleByte(first, 40, vector, vectorOffset, 5)
                + dotQ4_1PackedNibbleByte(second, 8, vector, vectorOffset, 9)
                + dotQ4_1PackedNibbleByte(second, 40, vector, vectorOffset, 13);
        float sum2 = dotQ4_1PackedNibbleByte(first, 16, vector, vectorOffset, 2)
                + dotQ4_1PackedNibbleByte(first, 48, vector, vectorOffset, 6)
                + dotQ4_1PackedNibbleByte(second, 16, vector, vectorOffset, 10)
                + dotQ4_1PackedNibbleByte(second, 48, vector, vectorOffset, 14);
        float sum3 = dotQ4_1PackedNibbleByte(first, 24, vector, vectorOffset, 3)
                + dotQ4_1PackedNibbleByte(first, 56, vector, vectorOffset, 7)
                + dotQ4_1PackedNibbleByte(second, 24, vector, vectorOffset, 11)
                + dotQ4_1PackedNibbleByte(second, 56, vector, vectorOffset, 15);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotQ4_1PackedNibbleByte(
            long packed,
            int shift,
            float[] vector,
            int vectorOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        return (quant & 0x0F) * vector[vectorOffset + index]
                + (quant >>> 4) * vector[vectorOffset + 16 + index];
    }

    private static float dotRowQ5_0UnscaledBlock(
            MemorySegment segment,
            long quantsOffset,
            int highBits,
            float[] vector,
            int vectorOffset) {
        long first = segment.get(LE_LONG, quantsOffset);
        long second = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        float sum0 = dotQ5_0PackedNibbleByte(first, 0, highBits, vector, vectorOffset, 0)
                + dotQ5_0PackedNibbleByte(first, 32, highBits, vector, vectorOffset, 4)
                + dotQ5_0PackedNibbleByte(second, 0, highBits, vector, vectorOffset, 8)
                + dotQ5_0PackedNibbleByte(second, 32, highBits, vector, vectorOffset, 12);
        float sum1 = dotQ5_0PackedNibbleByte(first, 8, highBits, vector, vectorOffset, 1)
                + dotQ5_0PackedNibbleByte(first, 40, highBits, vector, vectorOffset, 5)
                + dotQ5_0PackedNibbleByte(second, 8, highBits, vector, vectorOffset, 9)
                + dotQ5_0PackedNibbleByte(second, 40, highBits, vector, vectorOffset, 13);
        float sum2 = dotQ5_0PackedNibbleByte(first, 16, highBits, vector, vectorOffset, 2)
                + dotQ5_0PackedNibbleByte(first, 48, highBits, vector, vectorOffset, 6)
                + dotQ5_0PackedNibbleByte(second, 16, highBits, vector, vectorOffset, 10)
                + dotQ5_0PackedNibbleByte(second, 48, highBits, vector, vectorOffset, 14);
        float sum3 = dotQ5_0PackedNibbleByte(first, 24, highBits, vector, vectorOffset, 3)
                + dotQ5_0PackedNibbleByte(first, 56, highBits, vector, vectorOffset, 7)
                + dotQ5_0PackedNibbleByte(second, 24, highBits, vector, vectorOffset, 11)
                + dotQ5_0PackedNibbleByte(second, 56, highBits, vector, vectorOffset, 15);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotQ5_0PackedNibbleByte(
            long packed,
            int shift,
            int highBits,
            float[] vector,
            int vectorOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        int low = (quant & 0x0F) | (((highBits >>> index) & 1) << 4);
        int high = (quant >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
        return (low - 16) * vector[vectorOffset + index]
                + (high - 16) * vector[vectorOffset + 16 + index];
    }

    private static float dotRowQ5_1UnscaledBlock(
            MemorySegment segment,
            long quantsOffset,
            int highBits,
            float[] vector,
            int vectorOffset) {
        long first = segment.get(LE_LONG, quantsOffset);
        long second = segment.get(LE_LONG, quantsOffset + Long.BYTES);
        float sum0 = dotQ5_1PackedNibbleByte(first, 0, highBits, vector, vectorOffset, 0)
                + dotQ5_1PackedNibbleByte(first, 32, highBits, vector, vectorOffset, 4)
                + dotQ5_1PackedNibbleByte(second, 0, highBits, vector, vectorOffset, 8)
                + dotQ5_1PackedNibbleByte(second, 32, highBits, vector, vectorOffset, 12);
        float sum1 = dotQ5_1PackedNibbleByte(first, 8, highBits, vector, vectorOffset, 1)
                + dotQ5_1PackedNibbleByte(first, 40, highBits, vector, vectorOffset, 5)
                + dotQ5_1PackedNibbleByte(second, 8, highBits, vector, vectorOffset, 9)
                + dotQ5_1PackedNibbleByte(second, 40, highBits, vector, vectorOffset, 13);
        float sum2 = dotQ5_1PackedNibbleByte(first, 16, highBits, vector, vectorOffset, 2)
                + dotQ5_1PackedNibbleByte(first, 48, highBits, vector, vectorOffset, 6)
                + dotQ5_1PackedNibbleByte(second, 16, highBits, vector, vectorOffset, 10)
                + dotQ5_1PackedNibbleByte(second, 48, highBits, vector, vectorOffset, 14);
        float sum3 = dotQ5_1PackedNibbleByte(first, 24, highBits, vector, vectorOffset, 3)
                + dotQ5_1PackedNibbleByte(first, 56, highBits, vector, vectorOffset, 7)
                + dotQ5_1PackedNibbleByte(second, 24, highBits, vector, vectorOffset, 11)
                + dotQ5_1PackedNibbleByte(second, 56, highBits, vector, vectorOffset, 15);
        return sum0 + sum1 + sum2 + sum3;
    }

    private static float dotQ5_1PackedNibbleByte(
            long packed,
            int shift,
            int highBits,
            float[] vector,
            int vectorOffset,
            int index) {
        int quant = unsignedByte(packed, shift);
        int low = (quant & 0x0F) | (((highBits >>> index) & 1) << 4);
        int high = (quant >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
        return low * vector[vectorOffset + index]
                + high * vector[vectorOffset + 16 + index];
    }

}
