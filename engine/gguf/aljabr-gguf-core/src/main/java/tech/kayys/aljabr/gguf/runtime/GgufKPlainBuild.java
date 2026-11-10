package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.*;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.signedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorData.tensorData;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.remember;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.*;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q3KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q6KMatrix;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Prepared matrix builders for K-quant formats without group mins.
 * @author bhangun
 */
final class GgufKPlainBuild {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufKPlainBuild() {
    }

    static Q3KMatrix q3(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q3_K.id) {
            throw new IllegalArgumentException("Tensor is not Q3_K: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 16)];
        int[] unpackedScales = new int[16];

        long blockOffset = 0L;
        int groupBase = 0;
        int unpackedBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            fillQ3Block(source, blockOffset, groupScales, groupBase, quants, unpackedBase, unpackedScales);
            blockOffset += Q3_K_BLOCK_BYTES;
            groupBase += 16;
            unpackedBase += QK_K;
        }
        Q3KMatrix matrix = new Q3KMatrix(columns, rows, blocksPerRow, quants, groupScales);
        return remember(model, tensor, matrix);
    }

    static Q6KMatrix q6(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q6_K.id) {
            throw new IllegalArgumentException("Tensor is not Q6_K: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 16)];

        long blockOffset = 0L;
        int groupBase = 0;
        int unpackedBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            fillQ6Block(source, blockOffset, groupScales, groupBase, quants, unpackedBase);
            blockOffset += Q6_K_BLOCK_BYTES;
            groupBase += 16;
            unpackedBase += QK_K;
        }
        Q6KMatrix matrix = new Q6KMatrix(columns, rows, blocksPerRow, quants, groupScales);
        return remember(model, tensor, matrix);
    }

    private static void fillQ3Block(
            MemorySegment source,
            long blockOffset,
            float[] groupScales,
            int groupBase,
            byte[] quants,
            int unpackedBase,
            int[] unpackedScales) {
        float d = f16ToF32(source.get(LE_SHORT, blockOffset + 108));
        long hmaskOffset = blockOffset;
        long quantsOffset = blockOffset + 32;
        unpackQ3KScales(source, blockOffset + 96, unpackedScales);
        int scaleIndex = 0;
        int highMask = 1;
        long packedBase = quantsOffset;
        int superOutBase = unpackedBase;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            int outBase = superOutBase;
            int mask0 = highMask;
            int mask1 = mask0 << 1;
            int mask2 = mask1 << 1;
            int mask3 = mask2 << 1;
            for (int group = 0; group < 8; group++) {
                groupScales[groupBase + scaleIndex + group] = d * unpackedScales[scaleIndex + group];
            }
            unpackQ3KPreparedSuperBlock(
                    source,
                    hmaskOffset,
                    packedBase,
                    quants,
                    outBase,
                    mask0,
                    mask1,
                    mask2,
                    mask3);
            scaleIndex += 8;
            highMask <<= 4;
            packedBase += 32;
            superOutBase += 128;
        }
    }

    private static void fillQ6Block(
            MemorySegment source,
            long blockOffset,
            float[] groupScales,
            int groupBase,
            byte[] quants,
            int unpackedBase) {
        float d = f16ToF32(source.get(LE_SHORT, blockOffset + 208));
        long scalesOffset = blockOffset + 192;
        long scalesLow = source.get(LE_LONG, scalesOffset);
        long scalesHigh = source.get(LE_LONG, scalesOffset + Long.BYTES);
        for (int group = 0; group < 8; group++) {
            groupScales[groupBase + group] = d * signedByte(scalesLow, group * 8);
            groupScales[groupBase + 8 + group] = d * signedByte(scalesHigh, group * 8);
        }

        long lowBitsOffset = blockOffset;
        long highBitsOffset = blockOffset + 128;
        int outBase = unpackedBase;
        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            unpackQ6KPreparedSuperBlock(source, lowBitsOffset, highBitsOffset, quants, outBase);
            lowBitsOffset += 64;
            highBitsOffset += 32;
            outBase += 128;
        }
    }
}
