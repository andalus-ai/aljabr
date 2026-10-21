package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufByteValues.unsignedByte;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorData.tensorData;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.cachedKHasMins;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.remember;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q2_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.unpackQ2KPreparedSuperBlock;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q2KMatrix;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Prepared matrix builder for Q2_K tensors with optional group mins.
 * @author bhangun
 */
final class GgufQ2KBuild {
    private static final float[] EMPTY_FLOAT_ARRAY = new float[0];
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ2KBuild() {
    }

    static Q2KMatrix matrix(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q2_K.id) {
            throw new IllegalArgumentException("Tensor is not Q2_K: " + tensor.name());
        }
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 16)];
        Q2MinState minState = new Q2MinState();
        boolean mayHaveGroupMins = cachedKHasMins(model, tensor, totalBlocks, 16) != Boolean.FALSE;

        long blockOffset = 0L;
        int groupBase = 0;
        int unpackedBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            fillBlock(
                    source,
                    blockOffset,
                    groupScales,
                    minState,
                    mayHaveGroupMins,
                    groupBase,
                    quants,
                    unpackedBase);
            blockOffset += Q2_K_BLOCK_BYTES;
            groupBase += 16;
            unpackedBase += QK_K;
        }
        Q2KMatrix matrix = new Q2KMatrix(
                columns,
                rows,
                blocksPerRow,
                quants,
                groupScales,
                minState.hasGroupMins ? minState.groupMins : EMPTY_FLOAT_ARRAY,
                minState.hasGroupMins);
        return remember(model, tensor, matrix);
    }

    private static void fillBlock(
            MemorySegment source,
            long blockOffset,
            float[] groupScales,
            Q2MinState minState,
            boolean mayHaveGroupMins,
            int groupBase,
            byte[] quants,
            int unpackedBase) {
        float d = f16ToF32(source.get(LE_SHORT, blockOffset + 80));
        float dMin = mayHaveGroupMins ? f16ToF32(source.get(LE_SHORT, blockOffset + 82)) : 0.0f;
        long scalesOffset = blockOffset;
        long packedBase = blockOffset + 16;
        int scaleIndex = 0;
        int superOutBase = unpackedBase;

        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 128) {
            int outBase = superOutBase;
            long packedScales = source.get(LE_LONG, scalesOffset + scaleIndex);
            if (dMin != 0.0f) {
                fillScalePairsWithMins(groupScales, minState, groupBase + scaleIndex, d, dMin, packedScales);
            } else {
                fillScalePairs(groupScales, groupBase + scaleIndex, d, packedScales);
            }
            scaleIndex += 8;
            unpackQ2KPreparedSuperBlock(source, packedBase, quants, outBase);
            packedBase += 32;
            superOutBase += 128;
        }
    }

    private static void fillScalePairs(float[] groupScales, int groupBase, float d, long packedScales) {
        for (int pair = 0; pair < 4; pair++) {
            int scaleShift = pair * 16;
            int firstScale = unsignedByte(packedScales, scaleShift);
            int secondScale = unsignedByte(packedScales, scaleShift + 8);
            int groupIndex = groupBase + pair * 2;
            groupScales[groupIndex] = d * (firstScale & 0x0F);
            groupScales[groupIndex + 1] = d * (secondScale & 0x0F);
        }
    }

    private static void fillScalePairsWithMins(
            float[] groupScales,
            Q2MinState minState,
            int groupBase,
            float d,
            float dMin,
            long packedScales) {
        for (int pair = 0; pair < 4; pair++) {
            int scaleShift = pair * 16;
            int firstScale = unsignedByte(packedScales, scaleShift);
            int secondScale = unsignedByte(packedScales, scaleShift + 8);
            int groupIndex = groupBase + pair * 2;
            groupScales[groupIndex] = d * (firstScale & 0x0F);
            groupScales[groupIndex + 1] = d * (secondScale & 0x0F);
            fillMins(
                    groupScales,
                    minState,
                    groupIndex,
                    dMin,
                    firstScale >>> 4,
                    secondScale >>> 4);
        }
    }

    private static void fillMins(
            float[] groupScales,
            Q2MinState minState,
            int groupIndex,
            float dMin,
            int firstMinCode,
            int secondMinCode) {
        if (firstMinCode == 0 && secondMinCode == 0) {
            return;
        }
        if (minState.groupMins == null) {
            minState.groupMins = new float[groupScales.length];
        }
        minState.groupMins[groupIndex] = dMin * firstMinCode;
        minState.groupMins[groupIndex + 1] = dMin * secondMinCode;
        minState.hasGroupMins = true;
    }

    private static final class Q2MinState {
        private float[] groupMins;
        private boolean hasGroupMins;
    }
}
