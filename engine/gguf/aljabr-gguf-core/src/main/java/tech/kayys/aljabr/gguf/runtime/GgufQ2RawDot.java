package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q2_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUPS_PER_BLOCK;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUPS_PER_SUPER_BLOCK;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_GROUP_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_SUPER_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.minContribution;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw Q2_K row-dot kernels.
 *
 * <p>Q2_K packs eight 16-value groups per 128-value super block. The generic
 * path handles inline min correction, while specialized no-min and cached
 * group-sum paths remove that branch work for already-classified tensors.</p>
 * @author bhangun
 */
final class GgufQ2RawDot {
    private static final long HIGH_NIBBLE_MASK = 0xF0F0F0F0F0F0F0F0L;
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ2RawDot() {
    }

    static float dotRowQ2K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset) {
        return dotRowQ2K(segment, rowOffset, columns, vector, vectorOffset, null);
    }

    static float dotRowQ2K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        return dotRowQ2K(segment, rowOffset, columns, vector, 0, vectorGroupSums);
    }

    static float dotRowQ2K(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] vectorGroupSums) {
        if (vectorGroupSums != null) {
            return dotRowQ2KWithGroupSums(segment, rowOffset, columns, vector, vectorOffset, vectorGroupSums);
        }
        float sum = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int vectorGroupBase = vectorOffset / QK_GROUP_SIZE;
        boolean useCachedGroupSums = false;
        int blocks = columns / QK_K;
        for (int block = 0; block < blocks; block++) {
            float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 80));
            float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 82));
            boolean hasBlockMins = dMin != 0.0f;
            long scalesOffset = blockOffset;
            long packedBase = blockOffset + 16;
            int superVectorBase = vectorBase;
            int superGroupBase = vectorGroupBase;
            int scaleIndex = 0;

            for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += QK_SUPER_BLOCK_SIZE) {
                int groupBase = superGroupBase;
                long packedScales = segment.get(LE_LONG, scalesOffset + scaleIndex);
                boolean hasSuperMins = hasBlockMins && (packedScales & HIGH_NIBBLE_MASK) != 0L;
                int scale0 = unsignedByte(packedScales, 0);
                int scale1 = unsignedByte(packedScales, 8);
                int scale2 = unsignedByte(packedScales, 16);
                int scale3 = unsignedByte(packedScales, 24);
                int scale4 = unsignedByte(packedScales, 32);
                int scale5 = unsignedByte(packedScales, 40);
                int scale6 = unsignedByte(packedScales, 48);
                int scale7 = unsignedByte(packedScales, 56);
                float d0 = d * (scale0 & 0x0F);
                float d1 = d * (scale1 & 0x0F);
                float d2 = d * (scale2 & 0x0F);
                float d3 = d * (scale3 & 0x0F);
                float d4 = d * (scale4 & 0x0F);
                float d5 = d * (scale5 & 0x0F);
                float d6 = d * (scale6 & 0x0F);
                float d7 = d * (scale7 & 0x0F);
                int min0 = 0;
                int min1 = 0;
                int min2 = 0;
                int min3 = 0;
                int min4 = 0;
                int min5 = 0;
                int min6 = 0;
                int min7 = 0;
                boolean hasMin0 = false;
                boolean hasMin1 = false;
                boolean hasMin2 = false;
                boolean hasMin3 = false;
                boolean hasMin4 = false;
                boolean hasMin5 = false;
                boolean hasMin6 = false;
                boolean hasMin7 = false;
                if (hasSuperMins) {
                    min0 = scale0 >>> 4;
                    min1 = scale1 >>> 4;
                    min2 = scale2 >>> 4;
                    min3 = scale3 >>> 4;
                    min4 = scale4 >>> 4;
                    min5 = scale5 >>> 4;
                    min6 = scale6 >>> 4;
                    min7 = scale7 >>> 4;
                    hasMin0 = min0 != 0;
                    hasMin1 = min1 != 0;
                    hasMin2 = min2 != 0;
                    hasMin3 = min3 != 0;
                    hasMin4 = min4 != 0;
                    hasMin5 = min5 != 0;
                    hasMin6 = min6 != 0;
                    hasMin7 = min7 != 0;
                }
                float dot0 = 0.0f;
                float dot1 = 0.0f;
                float dot2 = 0.0f;
                float dot3 = 0.0f;
                float dot4 = 0.0f;
                float dot5 = 0.0f;
                float dot6 = 0.0f;
                float dot7 = 0.0f;
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;
                float sum4 = 0.0f;
                float sum5 = 0.0f;
                float sum6 = 0.0f;
                float sum7 = 0.0f;
                for (int i = 0; i < 16; i += Long.BYTES) {
                    long firstQuants = segment.get(LE_LONG, packedBase + i);
                    long secondQuants = segment.get(LE_LONG, packedBase + 16 + i);
                    int firstQuant0 = unsignedByte(firstQuants, 0);
                    int firstQuant1 = unsignedByte(firstQuants, 8);
                    int firstQuant2 = unsignedByte(firstQuants, 16);
                    int firstQuant3 = unsignedByte(firstQuants, 24);
                    int firstQuant4 = unsignedByte(firstQuants, 32);
                    int firstQuant5 = unsignedByte(firstQuants, 40);
                    int firstQuant6 = unsignedByte(firstQuants, 48);
                    int firstQuant7 = unsignedByte(firstQuants, 56);
                    int secondQuant0 = unsignedByte(secondQuants, 0);
                    int secondQuant1 = unsignedByte(secondQuants, 8);
                    int secondQuant2 = unsignedByte(secondQuants, 16);
                    int secondQuant3 = unsignedByte(secondQuants, 24);
                    int secondQuant4 = unsignedByte(secondQuants, 32);
                    int secondQuant5 = unsignedByte(secondQuants, 40);
                    int secondQuant6 = unsignedByte(secondQuants, 48);
                    int secondQuant7 = unsignedByte(secondQuants, 56);

                    float v00 = vector[superVectorBase + i];
                    float v01 = vector[superVectorBase + i + 1];
                    float v02 = vector[superVectorBase + i + 2];
                    float v03 = vector[superVectorBase + i + 3];
                    float v04 = vector[superVectorBase + i + 4];
                    float v05 = vector[superVectorBase + i + 5];
                    float v06 = vector[superVectorBase + i + 6];
                    float v07 = vector[superVectorBase + i + 7];
                    float v10 = vector[superVectorBase + 16 + i];
                    float v11 = vector[superVectorBase + 16 + i + 1];
                    float v12 = vector[superVectorBase + 16 + i + 2];
                    float v13 = vector[superVectorBase + 16 + i + 3];
                    float v14 = vector[superVectorBase + 16 + i + 4];
                    float v15 = vector[superVectorBase + 16 + i + 5];
                    float v16 = vector[superVectorBase + 16 + i + 6];
                    float v17 = vector[superVectorBase + 16 + i + 7];
                    float v20 = vector[superVectorBase + 32 + i];
                    float v21 = vector[superVectorBase + 32 + i + 1];
                    float v22 = vector[superVectorBase + 32 + i + 2];
                    float v23 = vector[superVectorBase + 32 + i + 3];
                    float v24 = vector[superVectorBase + 32 + i + 4];
                    float v25 = vector[superVectorBase + 32 + i + 5];
                    float v26 = vector[superVectorBase + 32 + i + 6];
                    float v27 = vector[superVectorBase + 32 + i + 7];
                    float v30 = vector[superVectorBase + 48 + i];
                    float v31 = vector[superVectorBase + 48 + i + 1];
                    float v32 = vector[superVectorBase + 48 + i + 2];
                    float v33 = vector[superVectorBase + 48 + i + 3];
                    float v34 = vector[superVectorBase + 48 + i + 4];
                    float v35 = vector[superVectorBase + 48 + i + 5];
                    float v36 = vector[superVectorBase + 48 + i + 6];
                    float v37 = vector[superVectorBase + 48 + i + 7];
                    float v40 = vector[superVectorBase + 64 + i];
                    float v41 = vector[superVectorBase + 64 + i + 1];
                    float v42 = vector[superVectorBase + 64 + i + 2];
                    float v43 = vector[superVectorBase + 64 + i + 3];
                    float v44 = vector[superVectorBase + 64 + i + 4];
                    float v45 = vector[superVectorBase + 64 + i + 5];
                    float v46 = vector[superVectorBase + 64 + i + 6];
                    float v47 = vector[superVectorBase + 64 + i + 7];
                    float v50 = vector[superVectorBase + 80 + i];
                    float v51 = vector[superVectorBase + 80 + i + 1];
                    float v52 = vector[superVectorBase + 80 + i + 2];
                    float v53 = vector[superVectorBase + 80 + i + 3];
                    float v54 = vector[superVectorBase + 80 + i + 4];
                    float v55 = vector[superVectorBase + 80 + i + 5];
                    float v56 = vector[superVectorBase + 80 + i + 6];
                    float v57 = vector[superVectorBase + 80 + i + 7];
                    float v60 = vector[superVectorBase + 96 + i];
                    float v61 = vector[superVectorBase + 96 + i + 1];
                    float v62 = vector[superVectorBase + 96 + i + 2];
                    float v63 = vector[superVectorBase + 96 + i + 3];
                    float v64 = vector[superVectorBase + 96 + i + 4];
                    float v65 = vector[superVectorBase + 96 + i + 5];
                    float v66 = vector[superVectorBase + 96 + i + 6];
                    float v67 = vector[superVectorBase + 96 + i + 7];
                    float v70 = vector[superVectorBase + 112 + i];
                    float v71 = vector[superVectorBase + 112 + i + 1];
                    float v72 = vector[superVectorBase + 112 + i + 2];
                    float v73 = vector[superVectorBase + 112 + i + 3];
                    float v74 = vector[superVectorBase + 112 + i + 4];
                    float v75 = vector[superVectorBase + 112 + i + 5];
                    float v76 = vector[superVectorBase + 112 + i + 6];
                    float v77 = vector[superVectorBase + 112 + i + 7];

                    dot0 += (firstQuant0 & 0x03) * v00
                            + (firstQuant1 & 0x03) * v01
                            + (firstQuant2 & 0x03) * v02
                            + (firstQuant3 & 0x03) * v03
                            + (firstQuant4 & 0x03) * v04
                            + (firstQuant5 & 0x03) * v05
                            + (firstQuant6 & 0x03) * v06
                            + (firstQuant7 & 0x03) * v07;
                    dot1 += (secondQuant0 & 0x03) * v10
                            + (secondQuant1 & 0x03) * v11
                            + (secondQuant2 & 0x03) * v12
                            + (secondQuant3 & 0x03) * v13
                            + (secondQuant4 & 0x03) * v14
                            + (secondQuant5 & 0x03) * v15
                            + (secondQuant6 & 0x03) * v16
                            + (secondQuant7 & 0x03) * v17;
                    dot2 += ((firstQuant0 >>> 2) & 0x03) * v20
                            + ((firstQuant1 >>> 2) & 0x03) * v21
                            + ((firstQuant2 >>> 2) & 0x03) * v22
                            + ((firstQuant3 >>> 2) & 0x03) * v23
                            + ((firstQuant4 >>> 2) & 0x03) * v24
                            + ((firstQuant5 >>> 2) & 0x03) * v25
                            + ((firstQuant6 >>> 2) & 0x03) * v26
                            + ((firstQuant7 >>> 2) & 0x03) * v27;
                    dot3 += ((secondQuant0 >>> 2) & 0x03) * v30
                            + ((secondQuant1 >>> 2) & 0x03) * v31
                            + ((secondQuant2 >>> 2) & 0x03) * v32
                            + ((secondQuant3 >>> 2) & 0x03) * v33
                            + ((secondQuant4 >>> 2) & 0x03) * v34
                            + ((secondQuant5 >>> 2) & 0x03) * v35
                            + ((secondQuant6 >>> 2) & 0x03) * v36
                            + ((secondQuant7 >>> 2) & 0x03) * v37;
                    dot4 += ((firstQuant0 >>> 4) & 0x03) * v40
                            + ((firstQuant1 >>> 4) & 0x03) * v41
                            + ((firstQuant2 >>> 4) & 0x03) * v42
                            + ((firstQuant3 >>> 4) & 0x03) * v43
                            + ((firstQuant4 >>> 4) & 0x03) * v44
                            + ((firstQuant5 >>> 4) & 0x03) * v45
                            + ((firstQuant6 >>> 4) & 0x03) * v46
                            + ((firstQuant7 >>> 4) & 0x03) * v47;
                    dot5 += ((secondQuant0 >>> 4) & 0x03) * v50
                            + ((secondQuant1 >>> 4) & 0x03) * v51
                            + ((secondQuant2 >>> 4) & 0x03) * v52
                            + ((secondQuant3 >>> 4) & 0x03) * v53
                            + ((secondQuant4 >>> 4) & 0x03) * v54
                            + ((secondQuant5 >>> 4) & 0x03) * v55
                            + ((secondQuant6 >>> 4) & 0x03) * v56
                            + ((secondQuant7 >>> 4) & 0x03) * v57;
                    dot6 += ((firstQuant0 >>> 6) & 0x03) * v60
                            + ((firstQuant1 >>> 6) & 0x03) * v61
                            + ((firstQuant2 >>> 6) & 0x03) * v62
                            + ((firstQuant3 >>> 6) & 0x03) * v63
                            + ((firstQuant4 >>> 6) & 0x03) * v64
                            + ((firstQuant5 >>> 6) & 0x03) * v65
                            + ((firstQuant6 >>> 6) & 0x03) * v66
                            + ((firstQuant7 >>> 6) & 0x03) * v67;
                    dot7 += ((secondQuant0 >>> 6) & 0x03) * v70
                            + ((secondQuant1 >>> 6) & 0x03) * v71
                            + ((secondQuant2 >>> 6) & 0x03) * v72
                            + ((secondQuant3 >>> 6) & 0x03) * v73
                            + ((secondQuant4 >>> 6) & 0x03) * v74
                            + ((secondQuant5 >>> 6) & 0x03) * v75
                            + ((secondQuant6 >>> 6) & 0x03) * v76
                            + ((secondQuant7 >>> 6) & 0x03) * v77;
                    if (!useCachedGroupSums) {
                        if (hasMin0) {
                            sum0 += v00 + v01 + v02 + v03 + v04 + v05 + v06 + v07;
                        }
                        if (hasMin1) {
                            sum1 += v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17;
                        }
                        if (hasMin2) {
                            sum2 += v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27;
                        }
                        if (hasMin3) {
                            sum3 += v30 + v31 + v32 + v33 + v34 + v35 + v36 + v37;
                        }
                        if (hasMin4) {
                            sum4 += v40 + v41 + v42 + v43 + v44 + v45 + v46 + v47;
                        }
                        if (hasMin5) {
                            sum5 += v50 + v51 + v52 + v53 + v54 + v55 + v56 + v57;
                        }
                        if (hasMin6) {
                            sum6 += v60 + v61 + v62 + v63 + v64 + v65 + v66 + v67;
                        }
                        if (hasMin7) {
                            sum7 += v70 + v71 + v72 + v73 + v74 + v75 + v76 + v77;
                        }
                    }
                }
                sum += d0 * dot0 + d1 * dot1 + d2 * dot2 + d3 * dot3
                        + d4 * dot4 + d5 * dot5 + d6 * dot6 + d7 * dot7;
                if (!useCachedGroupSums) {
                    if (hasMin0) {
                        sum -= dMin * min0 * sum0;
                    }
                    if (hasMin1) {
                        sum -= dMin * min1 * sum1;
                    }
                    if (hasMin2) {
                        sum -= dMin * min2 * sum2;
                    }
                    if (hasMin3) {
                        sum -= dMin * min3 * sum3;
                    }
                    if (hasMin4) {
                        sum -= dMin * min4 * sum4;
                    }
                    if (hasMin5) {
                        sum -= dMin * min5 * sum5;
                    }
                    if (hasMin6) {
                        sum -= dMin * min6 * sum6;
                    }
                    if (hasMin7) {
                        sum -= dMin * min7 * sum7;
                    }
                } else if (hasSuperMins) {
                    sum -= minContribution(dMin, min0) * vectorGroupSums[groupBase]
                            + minContribution(dMin, min1) * vectorGroupSums[groupBase + 1]
                            + minContribution(dMin, min2) * vectorGroupSums[groupBase + 2]
                            + minContribution(dMin, min3) * vectorGroupSums[groupBase + 3]
                            + minContribution(dMin, min4) * vectorGroupSums[groupBase + 4]
                            + minContribution(dMin, min5) * vectorGroupSums[groupBase + 5]
                            + minContribution(dMin, min6) * vectorGroupSums[groupBase + 6]
                            + minContribution(dMin, min7) * vectorGroupSums[groupBase + 7];
                }
                scaleIndex += 8;
                packedBase += 32;
                superVectorBase += QK_SUPER_BLOCK_SIZE;
                superGroupBase += QK_GROUPS_PER_SUPER_BLOCK;
            }
            blockOffset += Q2_K_BLOCK_BYTES;
            vectorBase += QK_K;
            vectorGroupBase += QK_GROUPS_PER_BLOCK;
        }
        return sum;
    }

    static float dotRowQ2KWithGroupSums(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            float[] vectorGroupSums) {
        return dotRowQ2KWithGroupSums(segment, rowOffset, columns, vector, 0, vectorGroupSums);
    }

    private static float dotRowQ2KWithGroupSums(
            MemorySegment segment,
            long rowOffset,
            int columns,
            float[] vector,
            int vectorOffset,
            float[] vectorGroupSums) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        long blockOffset = rowOffset;
        int vectorBase = vectorOffset;
        int groupIndex = vectorOffset / QK_GROUP_SIZE;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            sum0 += dotRowQ2KGroupSumBlock(segment, blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            sum1 += dotRowQ2KGroupSumBlock(
                    segment,
                    blockOffset + Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS_PER_BLOCK);
            sum2 += dotRowQ2KGroupSumBlock(
                    segment,
                    blockOffset + 2L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS_PER_BLOCK);
            sum3 += dotRowQ2KGroupSumBlock(
                    segment,
                    blockOffset + 3L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS_PER_BLOCK);
            blockOffset += 4L * Q2_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
            groupIndex += 4 * QK_GROUPS_PER_BLOCK;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ2KGroupSumBlock(segment, blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            blockOffset += Q2_K_BLOCK_BYTES;
            vectorBase += QK_K;
            groupIndex += QK_GROUPS_PER_BLOCK;
        }
        return sum;
    }

    static void dotRowsQ2KWithGroupSums4(
            MemorySegment segment,
            long rowOffset,
            long rowBytes,
            int columns,
            float[] vector,
            float[] vectorGroupSums,
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
        int groupIndex = 0;
        int blocks = columns / QK_K;
        int block = 0;
        int unrolledLimit = blocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            row0Sum0 += dotRowQ2KGroupSumBlock(segment, row0Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row1Sum0 += dotRowQ2KGroupSumBlock(segment, row1Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row2Sum0 += dotRowQ2KGroupSumBlock(segment, row2Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row3Sum0 += dotRowQ2KGroupSumBlock(segment, row3Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row0Sum1 += dotRowQ2KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS_PER_BLOCK);
            row1Sum1 += dotRowQ2KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS_PER_BLOCK);
            row2Sum1 += dotRowQ2KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS_PER_BLOCK);
            row3Sum1 += dotRowQ2KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K,
                    vectorGroupSums,
                    groupIndex + QK_GROUPS_PER_BLOCK);
            row0Sum2 += dotRowQ2KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS_PER_BLOCK);
            row1Sum2 += dotRowQ2KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS_PER_BLOCK);
            row2Sum2 += dotRowQ2KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS_PER_BLOCK);
            row3Sum2 += dotRowQ2KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K,
                    vectorGroupSums,
                    groupIndex + 2 * QK_GROUPS_PER_BLOCK);
            row0Sum3 += dotRowQ2KGroupSumBlock(
                    segment,
                    row0Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS_PER_BLOCK);
            row1Sum3 += dotRowQ2KGroupSumBlock(
                    segment,
                    row1Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS_PER_BLOCK);
            row2Sum3 += dotRowQ2KGroupSumBlock(
                    segment,
                    row2Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS_PER_BLOCK);
            row3Sum3 += dotRowQ2KGroupSumBlock(
                    segment,
                    row3Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K,
                    vectorGroupSums,
                    groupIndex + 3 * QK_GROUPS_PER_BLOCK);
            blockOffset += 4L * Q2_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
            groupIndex += 4 * QK_GROUPS_PER_BLOCK;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ2KGroupSumBlock(segment, row0Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row1Sum += dotRowQ2KGroupSumBlock(segment, row1Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row2Sum += dotRowQ2KGroupSumBlock(segment, row2Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            row3Sum += dotRowQ2KGroupSumBlock(segment, row3Offset + blockOffset, vector, vectorBase, vectorGroupSums, groupIndex);
            blockOffset += Q2_K_BLOCK_BYTES;
            vectorBase += QK_K;
            groupIndex += QK_GROUPS_PER_BLOCK;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowQ2KGroupSumBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase,
            float[] vectorGroupSums,
            int groupIndex) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 80));
        long packedBase = blockOffset + 16;
        long firstScales = segment.get(LE_LONG, blockOffset);
        long secondScales = segment.get(LE_LONG, blockOffset + Long.BYTES);
        float sum = dotRowQ2KNoMinSuperBlock(segment, packedBase, vector, vectorBase, d, firstScales)
                + dotRowQ2KNoMinSuperBlock(segment, packedBase + 32, vector, vectorBase + 128, d, secondScales);
        float dMin = f16ToF32(segment.get(LE_SHORT, blockOffset + 82));
        if (dMin != 0.0f) {
            sum -= q2KMinContribution(dMin, firstScales, vectorGroupSums, groupIndex)
                    + q2KMinContribution(dMin, secondScales, vectorGroupSums, groupIndex + QK_GROUPS_PER_SUPER_BLOCK);
        }
        return sum;
    }

    private static float q2KMinContribution(
            float dMin,
            long packedScales,
            float[] vectorGroupSums,
            int groupIndex) {
        if ((packedScales & HIGH_NIBBLE_MASK) == 0L) {
            return 0.0f;
        }
        float correction0 = (dMin * (unsignedByte(packedScales, 0) >>> 4)) * vectorGroupSums[groupIndex]
                + (dMin * (unsignedByte(packedScales, 8) >>> 4)) * vectorGroupSums[groupIndex + 1];
        float correction1 = (dMin * (unsignedByte(packedScales, 16) >>> 4)) * vectorGroupSums[groupIndex + 2]
                + (dMin * (unsignedByte(packedScales, 24) >>> 4)) * vectorGroupSums[groupIndex + 3];
        float correction2 = (dMin * (unsignedByte(packedScales, 32) >>> 4)) * vectorGroupSums[groupIndex + 4]
                + (dMin * (unsignedByte(packedScales, 40) >>> 4)) * vectorGroupSums[groupIndex + 5];
        float correction3 = (dMin * (unsignedByte(packedScales, 48) >>> 4)) * vectorGroupSums[groupIndex + 6]
                + (dMin * (unsignedByte(packedScales, 56) >>> 4)) * vectorGroupSums[groupIndex + 7];
        return correction0 + correction1 + correction2 + correction3;
    }

    static float dotRowQ2KNoMins(
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
            sum0 += dotRowQ2KNoMinBlock(segment, blockOffset, vector, vectorBase);
            sum1 += dotRowQ2KNoMinBlock(
                    segment,
                    blockOffset + Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + QK_K);
            sum2 += dotRowQ2KNoMinBlock(
                    segment,
                    blockOffset + 2L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 2 * QK_K);
            sum3 += dotRowQ2KNoMinBlock(
                    segment,
                    blockOffset + 3L * Q2_K_BLOCK_BYTES,
                    vector,
                    vectorBase + 3 * QK_K);
            blockOffset += 4L * Q2_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; block < blocks; block++) {
            sum += dotRowQ2KNoMinBlock(segment, blockOffset, vector, vectorBase);
            blockOffset += Q2_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        return sum;
    }

    static void dotRowsQ2KNoMins4(
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
            row0Sum0 += dotRowQ2KNoMinBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum0 += dotRowQ2KNoMinBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum0 += dotRowQ2KNoMinBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum0 += dotRowQ2KNoMinBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            row0Sum1 += dotRowQ2KNoMinBlock(segment, row0Offset + blockOffset + Q2_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row1Sum1 += dotRowQ2KNoMinBlock(segment, row1Offset + blockOffset + Q2_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row2Sum1 += dotRowQ2KNoMinBlock(segment, row2Offset + blockOffset + Q2_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row3Sum1 += dotRowQ2KNoMinBlock(segment, row3Offset + blockOffset + Q2_K_BLOCK_BYTES, vector, vectorBase + QK_K);
            row0Sum2 += dotRowQ2KNoMinBlock(segment, row0Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row1Sum2 += dotRowQ2KNoMinBlock(segment, row1Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row2Sum2 += dotRowQ2KNoMinBlock(segment, row2Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row3Sum2 += dotRowQ2KNoMinBlock(segment, row3Offset + blockOffset + 2L * Q2_K_BLOCK_BYTES, vector, vectorBase + 2 * QK_K);
            row0Sum3 += dotRowQ2KNoMinBlock(segment, row0Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            row1Sum3 += dotRowQ2KNoMinBlock(segment, row1Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            row2Sum3 += dotRowQ2KNoMinBlock(segment, row2Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            row3Sum3 += dotRowQ2KNoMinBlock(segment, row3Offset + blockOffset + 3L * Q2_K_BLOCK_BYTES, vector, vectorBase + 3 * QK_K);
            blockOffset += 4L * Q2_K_BLOCK_BYTES;
            vectorBase += 4 * QK_K;
        }
        float row0Sum = row0Sum0 + row0Sum1 + row0Sum2 + row0Sum3;
        float row1Sum = row1Sum0 + row1Sum1 + row1Sum2 + row1Sum3;
        float row2Sum = row2Sum0 + row2Sum1 + row2Sum2 + row2Sum3;
        float row3Sum = row3Sum0 + row3Sum1 + row3Sum2 + row3Sum3;
        for (; block < blocks; block++) {
            row0Sum += dotRowQ2KNoMinBlock(segment, row0Offset + blockOffset, vector, vectorBase);
            row1Sum += dotRowQ2KNoMinBlock(segment, row1Offset + blockOffset, vector, vectorBase);
            row2Sum += dotRowQ2KNoMinBlock(segment, row2Offset + blockOffset, vector, vectorBase);
            row3Sum += dotRowQ2KNoMinBlock(segment, row3Offset + blockOffset, vector, vectorBase);
            blockOffset += Q2_K_BLOCK_BYTES;
            vectorBase += QK_K;
        }
        output[outputOffset] = row0Sum;
        output[outputOffset + 1] = row1Sum;
        output[outputOffset + 2] = row2Sum;
        output[outputOffset + 3] = row3Sum;
    }

    private static float dotRowQ2KNoMinBlock(
            MemorySegment segment,
            long blockOffset,
            float[] vector,
            int vectorBase) {
        float d = f16ToF32(segment.get(LE_SHORT, blockOffset + 80));
        long scalesOffset = blockOffset;
        long packedBase = blockOffset + 16;
        return dotRowQ2KNoMinSuperBlock(
                segment,
                packedBase,
                vector,
                vectorBase,
                d,
                segment.get(LE_LONG, scalesOffset))
                + dotRowQ2KNoMinSuperBlock(
                        segment,
                        packedBase + 32,
                        vector,
                        vectorBase + 128,
                        d,
                        segment.get(LE_LONG, scalesOffset + Long.BYTES));
    }

    private static float dotRowQ2KNoMinSuperBlock(
            MemorySegment segment,
            long packedBase,
            float[] vector,
            int vectorBase,
            float d,
            long packedScales) {
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        float dot4 = 0.0f;
        float dot5 = 0.0f;
        float dot6 = 0.0f;
        float dot7 = 0.0f;
        for (int i = 0; i < 16; i += Long.BYTES) {
            long firstQuants = segment.get(LE_LONG, packedBase + i);
            long secondQuants = segment.get(LE_LONG, packedBase + 16 + i);
            int firstQuant0 = unsignedByte(firstQuants, 0);
            int firstQuant1 = unsignedByte(firstQuants, 8);
            int firstQuant2 = unsignedByte(firstQuants, 16);
            int firstQuant3 = unsignedByte(firstQuants, 24);
            int firstQuant4 = unsignedByte(firstQuants, 32);
            int firstQuant5 = unsignedByte(firstQuants, 40);
            int firstQuant6 = unsignedByte(firstQuants, 48);
            int firstQuant7 = unsignedByte(firstQuants, 56);
            int secondQuant0 = unsignedByte(secondQuants, 0);
            int secondQuant1 = unsignedByte(secondQuants, 8);
            int secondQuant2 = unsignedByte(secondQuants, 16);
            int secondQuant3 = unsignedByte(secondQuants, 24);
            int secondQuant4 = unsignedByte(secondQuants, 32);
            int secondQuant5 = unsignedByte(secondQuants, 40);
            int secondQuant6 = unsignedByte(secondQuants, 48);
            int secondQuant7 = unsignedByte(secondQuants, 56);
            dot0 += (firstQuant0 & 0x03) * vector[vectorBase + i]
                    + (firstQuant1 & 0x03) * vector[vectorBase + i + 1]
                    + (firstQuant2 & 0x03) * vector[vectorBase + i + 2]
                    + (firstQuant3 & 0x03) * vector[vectorBase + i + 3]
                    + (firstQuant4 & 0x03) * vector[vectorBase + i + 4]
                    + (firstQuant5 & 0x03) * vector[vectorBase + i + 5]
                    + (firstQuant6 & 0x03) * vector[vectorBase + i + 6]
                    + (firstQuant7 & 0x03) * vector[vectorBase + i + 7];
            dot1 += (secondQuant0 & 0x03) * vector[vectorBase + 16 + i]
                    + (secondQuant1 & 0x03) * vector[vectorBase + 16 + i + 1]
                    + (secondQuant2 & 0x03) * vector[vectorBase + 16 + i + 2]
                    + (secondQuant3 & 0x03) * vector[vectorBase + 16 + i + 3]
                    + (secondQuant4 & 0x03) * vector[vectorBase + 16 + i + 4]
                    + (secondQuant5 & 0x03) * vector[vectorBase + 16 + i + 5]
                    + (secondQuant6 & 0x03) * vector[vectorBase + 16 + i + 6]
                    + (secondQuant7 & 0x03) * vector[vectorBase + 16 + i + 7];
            dot2 += ((firstQuant0 >>> 2) & 0x03) * vector[vectorBase + 32 + i]
                    + ((firstQuant1 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 1]
                    + ((firstQuant2 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 2]
                    + ((firstQuant3 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 3]
                    + ((firstQuant4 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 4]
                    + ((firstQuant5 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 5]
                    + ((firstQuant6 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 6]
                    + ((firstQuant7 >>> 2) & 0x03) * vector[vectorBase + 32 + i + 7];
            dot3 += ((secondQuant0 >>> 2) & 0x03) * vector[vectorBase + 48 + i]
                    + ((secondQuant1 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 1]
                    + ((secondQuant2 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 2]
                    + ((secondQuant3 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 3]
                    + ((secondQuant4 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 4]
                    + ((secondQuant5 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 5]
                    + ((secondQuant6 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 6]
                    + ((secondQuant7 >>> 2) & 0x03) * vector[vectorBase + 48 + i + 7];
            dot4 += ((firstQuant0 >>> 4) & 0x03) * vector[vectorBase + 64 + i]
                    + ((firstQuant1 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 1]
                    + ((firstQuant2 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 2]
                    + ((firstQuant3 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 3]
                    + ((firstQuant4 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 4]
                    + ((firstQuant5 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 5]
                    + ((firstQuant6 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 6]
                    + ((firstQuant7 >>> 4) & 0x03) * vector[vectorBase + 64 + i + 7];
            dot5 += ((secondQuant0 >>> 4) & 0x03) * vector[vectorBase + 80 + i]
                    + ((secondQuant1 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 1]
                    + ((secondQuant2 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 2]
                    + ((secondQuant3 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 3]
                    + ((secondQuant4 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 4]
                    + ((secondQuant5 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 5]
                    + ((secondQuant6 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 6]
                    + ((secondQuant7 >>> 4) & 0x03) * vector[vectorBase + 80 + i + 7];
            dot6 += ((firstQuant0 >>> 6) & 0x03) * vector[vectorBase + 96 + i]
                    + ((firstQuant1 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 1]
                    + ((firstQuant2 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 2]
                    + ((firstQuant3 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 3]
                    + ((firstQuant4 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 4]
                    + ((firstQuant5 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 5]
                    + ((firstQuant6 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 6]
                    + ((firstQuant7 >>> 6) & 0x03) * vector[vectorBase + 96 + i + 7];
            dot7 += ((secondQuant0 >>> 6) & 0x03) * vector[vectorBase + 112 + i]
                    + ((secondQuant1 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 1]
                    + ((secondQuant2 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 2]
                    + ((secondQuant3 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 3]
                    + ((secondQuant4 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 4]
                    + ((secondQuant5 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 5]
                    + ((secondQuant6 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 6]
                    + ((secondQuant7 >>> 6) & 0x03) * vector[vectorBase + 112 + i + 7];
        }
        return d * ((unsignedByte(packedScales, 0) & 0x0F) * dot0
                + (unsignedByte(packedScales, 8) & 0x0F) * dot1
                + (unsignedByte(packedScales, 16) & 0x0F) * dot2
                + (unsignedByte(packedScales, 24) & 0x0F) * dot3
                + (unsignedByte(packedScales, 32) & 0x0F) * dot4
                + (unsignedByte(packedScales, 40) & 0x0F) * dot5
                + (unsignedByte(packedScales, 48) & 0x0F) * dot6
                + (unsignedByte(packedScales, 56) & 0x0F) * dot7);
    }
}
