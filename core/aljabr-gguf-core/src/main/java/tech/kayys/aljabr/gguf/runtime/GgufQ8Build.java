package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.*;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.*;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorData.tensorData;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.remember;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKQuantLayout.iq4XSScalePacked;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.*;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Prepared matrix builder for Q8-family, nibble, and ternary GGUF formats.
 * @author bhangun
 */
final class GgufQ8Build {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ8Build() {
    }

    static Q8Matrix matrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        int typeId = tensor.typeId();
        if (!usesQ8PreparedCache(typeId)) {
            throw new IllegalArgumentException("Tensor is not a supported Q8 prepared type: " + tensor.name());
        }
        if (typeId == GgmlType.NVFP4.id) {
            return nvfp4Matrix(model, tensor);
        }
        if (typeId == GgmlType.IQ4_XS.id) {
            return iq4XSMatrix(model, tensor);
        }
        return GgufQ8RegBuild.matrix(model, tensor, typeId);
    }

    private static Q8Matrix nvfp4Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int groupsPerRow = columns / NVFP4_SUB_BLOCK_SIZE;
        int blocksPerRow = columns / NVFP4_BLOCK_SIZE;
        int totalGroups = Math.multiplyExact(rows, groupsPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalGroups, NVFP4_SUB_BLOCK_SIZE)];
        float[] blockScales = new float[totalGroups];

        long sourceOffset = 0L;
        int matrixGroup = 0;
        int qBase = 0;
        for (int row = 0; row < rows; row++) {
            for (int block = 0; block < blocksPerRow; block++) {
                int packedScales = source.get(LE_INT, sourceOffset);
                long subQuantsOffset = sourceOffset + NVFP4_SUB_BLOCKS;

                for (int sub = 0; sub < NVFP4_SUB_BLOCKS; sub++) {
                    blockScales[matrixGroup] = ue4m3ToF32(unsignedByte(packedScales, sub * 8));
                    unpackNVFP4SubBlockPrepared(source, subQuantsOffset, quants, qBase);
                    matrixGroup++;
                    qBase += NVFP4_SUB_BLOCK_SIZE;
                    subQuantsOffset += NVFP4_SUB_BLOCK_SIZE / 2;
                }
                sourceOffset += NVFP4_BLOCK_BYTES;
            }
        }
        Q8Matrix matrix = new Q8Matrix(columns, rows, groupsPerRow, NVFP4_SUB_BLOCK_SIZE, quants, blockScales);
        return remember(model, tensor, matrix);
    }

    private static Q8Matrix iq4XSMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int groupsPerRow = columns / IQ4_XS_GROUP_SIZE;
        int kBlocksPerRow = columns / QK_K;
        int totalGroups = Math.multiplyExact(rows, groupsPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalGroups, IQ4_XS_GROUP_SIZE)];
        float[] blockScales = new float[totalGroups];

        long sourceOffset = 0L;
        int matrixGroup = 0;
        int qBase = 0;
        for (int row = 0; row < rows; row++) {
            for (int kBlock = 0; kBlock < kBlocksPerRow; kBlock++) {
                float d = f16ToF32(source.get(LE_SHORT, sourceOffset));
                int scalesH = source.get(LE_SHORT, sourceOffset + 2) & 0xFFFF;
                int scalesL = source.get(LE_INT, sourceOffset + 4);
                long groupQuantsOffset = sourceOffset + 8;

                for (int group = 0; group < IQ4_XS_GROUPS; group++) {
                    blockScales[matrixGroup] = d * (iq4XSScalePacked(scalesH, scalesL, group) - 32);
                    unpackIQ4NLPrepared(source, groupQuantsOffset, quants, qBase);
                    matrixGroup++;
                    qBase += IQ4_XS_GROUP_SIZE;
                    groupQuantsOffset += IQ4_XS_GROUP_SIZE / 2;
                }
                sourceOffset += IQ4_XS_BLOCK_BYTES;
            }
        }
        Q8Matrix matrix = new Q8Matrix(columns, rows, groupsPerRow, IQ4_XS_GROUP_SIZE, quants, blockScales);
        return remember(model, tensor, matrix);
    }
}
