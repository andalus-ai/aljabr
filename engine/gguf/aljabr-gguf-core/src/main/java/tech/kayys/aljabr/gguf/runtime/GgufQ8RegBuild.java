package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.*;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.u8;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorData.tensorData;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.remember;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.*;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Prepared matrix builder for regular one-block-per-scale Q8-family formats.
 * @author bhangun
 */
final class GgufQ8RegBuild {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ8RegBuild() {
    }

    static Q8Matrix matrix(GGUFModel model, GGUFTensorInfo tensor, int typeId) {
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blockSize = q8BlockSize(typeId);
        int blockBytes = q8BlockBytes(typeId);
        int blocksPerRow = columns / blockSize;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, blockSize)];
        float[] blockScales = new float[totalBlocks];

        fillBlocks(source, typeId, totalBlocks, blockSize, blockBytes, blockScales, quants);
        Q8Matrix matrix = new Q8Matrix(columns, rows, blocksPerRow, blockSize, quants, blockScales);
        return remember(model, tensor, matrix);
    }

    private static void fillBlocks(
            MemorySegment source,
            int typeId,
            int totalBlocks,
            int blockSize,
            int blockBytes,
            float[] blockScales,
            byte[] quants) {
        if (typeId == GgmlType.Q8_0.id) {
            fillF16CopyBlocks(source, totalBlocks, Q8_0_BLOCK_BYTES, Short.BYTES, blockSize, blockScales, quants);
        } else if (typeId == GgmlType.Q8_1.id) {
            fillF16CopyBlocks(source, totalBlocks, Q8_1_BLOCK_BYTES, 2 * Short.BYTES, blockSize, blockScales, quants);
        } else if (typeId == GgmlType.Q8_K.id) {
            fillQ8KBlocks(source, totalBlocks, blockScales, quants);
        } else if (typeId == GgmlType.Q1_0.id) {
            fillQ1Blocks(source, totalBlocks, blockScales, quants);
        } else if (typeId == GgmlType.TQ1_0.id) {
            fillTQ1Blocks(source, totalBlocks, blockScales, quants);
        } else if (typeId == GgmlType.TQ2_0.id) {
            fillTQ2Blocks(source, totalBlocks, blockScales, quants);
        } else if (typeId == GgmlType.MXFP4.id) {
            fillMXFP4Blocks(source, totalBlocks, blockScales, quants);
        } else {
            fillIQ4NLBlocks(source, totalBlocks, blockSize, blockBytes, blockScales, quants);
        }
    }

    private static void fillF16CopyBlocks(
            MemorySegment source,
            int totalBlocks,
            int blockBytes,
            int quantOffset,
            int blockSize,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            copyPreparedQuants(source, sourceOffset + quantOffset, quants, qBase, blockSize);
            sourceOffset += blockBytes;
            qBase += blockSize;
        }
    }

    private static void fillQ8KBlocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = source.get(LE_FLOAT, sourceOffset);
            copyPreparedQuants(source, sourceOffset + Float.BYTES, quants, qBase, QK_K);
            sourceOffset += Q8_K_BLOCK_BYTES;
            qBase += QK_K;
        }
    }

    private static void fillQ1Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            unpackQ1_0Prepared(source, sourceOffset + Short.BYTES, quants, qBase);
            sourceOffset += Q1_0_BLOCK_BYTES;
            qBase += Q1_0_BLOCK_SIZE;
        }
    }

    private static void fillTQ1Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset + TQ1_0_SCALE_OFFSET));
            unpackTQ1_0Prepared(source, sourceOffset, quants, qBase);
            sourceOffset += TQ1_0_BLOCK_BYTES;
            qBase += TQ1_0_BLOCK_SIZE;
        }
    }

    private static void fillTQ2Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset + TQ2_0_QUANT_BYTES));
            unpackTQ2_0Prepared(source, sourceOffset, quants, qBase);
            sourceOffset += TQ2_0_BLOCK_BYTES;
            qBase += TQ2_0_BLOCK_SIZE;
        }
    }

    private static void fillMXFP4Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = e8m0ToF32Half(u8(source, sourceOffset));
            unpackMXFP4Prepared(source, sourceOffset + 1, quants, qBase);
            sourceOffset += MXFP4_BLOCK_BYTES;
            qBase += MXFP4_BLOCK_SIZE;
        }
    }

    private static void fillIQ4NLBlocks(
            MemorySegment source,
            int totalBlocks,
            int blockSize,
            int blockBytes,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            unpackIQ4NLPrepared(source, sourceOffset + Short.BYTES, quants, qBase);
            sourceOffset += blockBytes;
            qBase += blockSize;
        }
    }
}
