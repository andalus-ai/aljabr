package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorData.tensorData;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.remember;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.*;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q32Matrix;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Prepared matrix builder for 32-value legacy quant blocks.
 * @author bhangun
 */
final class GgufQ32Build {
    private static final float[] EMPTY_FLOAT_ARRAY = new float[0];
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ32Build() {
    }

    static Q32Matrix matrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (!supportsQ32PreparedType(tensor.typeId())) {
            throw new IllegalArgumentException("Tensor is not a supported 32-value quant type: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / Q4_0_BLOCK_SIZE;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, Q4_0_BLOCK_SIZE)];
        float[] blockScales = new float[totalBlocks];
        int typeId = tensor.typeId();
        boolean typeHasBlockBiases = q32PreparedHasBlockBiases(typeId);
        float[] blockBiases = typeHasBlockBiases ? new float[totalBlocks] : EMPTY_FLOAT_ARRAY;

        boolean hasBlockBiases = fillBlocks(source, typeId, totalBlocks, blockScales, blockBiases, quants);
        if (!hasBlockBiases) {
            blockBiases = EMPTY_FLOAT_ARRAY;
        }
        Q32Matrix matrix = new Q32Matrix(columns, rows, blocksPerRow, quants, blockScales, blockBiases, hasBlockBiases);
        return remember(model, tensor, matrix);
    }

    private static boolean fillBlocks(
            MemorySegment source,
            int typeId,
            int totalBlocks,
            float[] blockScales,
            float[] blockBiases,
            byte[] quants) {
        if (typeId == GgmlType.Q4_0.id) {
            fillQ4_0Blocks(source, totalBlocks, blockScales, quants);
            return false;
        } else if (typeId == GgmlType.Q4_1.id) {
            return fillQ4_1Blocks(source, totalBlocks, blockScales, blockBiases, quants);
        } else if (typeId == GgmlType.Q5_0.id) {
            fillQ5_0Blocks(source, totalBlocks, blockScales, quants);
            return false;
        }
        return fillQ5_1Blocks(source, totalBlocks, blockScales, blockBiases, quants);
    }

    private static void fillQ4_0Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            unpackQ4_0Prepared(source, sourceOffset + Short.BYTES, quants, qBase);
            sourceOffset += Q4_0_BLOCK_BYTES;
            qBase += Q4_0_BLOCK_SIZE;
        }
    }

    private static boolean fillQ4_1Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            float[] blockBiases,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        boolean hasBlockBiases = false;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            blockBiases[block] = f16ToF32(source.get(LE_SHORT, sourceOffset + Short.BYTES));
            hasBlockBiases |= blockBiases[block] != 0.0f;
            unpackQ4_1Prepared(source, sourceOffset + 2 * Short.BYTES, quants, qBase);
            sourceOffset += Q4_1_BLOCK_BYTES;
            qBase += Q4_0_BLOCK_SIZE;
        }
        return hasBlockBiases;
    }

    private static void fillQ5_0Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            int highBits = source.get(LE_INT, sourceOffset + 2);
            unpackQ5_0Prepared(source, sourceOffset + 6, highBits, quants, qBase);
            sourceOffset += Q5_0_BLOCK_BYTES;
            qBase += Q4_0_BLOCK_SIZE;
        }
    }

    private static boolean fillQ5_1Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] blockScales,
            float[] blockBiases,
            byte[] quants) {
        long sourceOffset = 0L;
        int qBase = 0;
        boolean hasBlockBiases = false;
        for (int block = 0; block < totalBlocks; block++) {
            blockScales[block] = f16ToF32(source.get(LE_SHORT, sourceOffset));
            blockBiases[block] = f16ToF32(source.get(LE_SHORT, sourceOffset + Short.BYTES));
            hasBlockBiases |= blockBiases[block] != 0.0f;
            int highBits = source.get(LE_INT, sourceOffset + 2 * Short.BYTES);
            unpackQ5_1Prepared(source, sourceOffset + 2 * Short.BYTES + Integer.BYTES, highBits, quants, qBase);
            sourceOffset += Q5_1_BLOCK_BYTES;
            qBase += Q4_0_BLOCK_SIZE;
        }
        return hasBlockBiases;
    }
}
