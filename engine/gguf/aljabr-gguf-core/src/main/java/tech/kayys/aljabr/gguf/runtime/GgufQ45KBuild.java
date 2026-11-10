package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorData.tensorData;
import static tech.kayys.aljabr.gguf.runtime.GgufTensorShape.*;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.cachedKHasMins;
import static tech.kayys.aljabr.gguf.runtime.GgufPreparedMatrixEstimator.remember;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.unpackQ4KPreparedGroup;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantUnpacker.unpackQ5KPreparedGroup;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q5KMatrix;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Prepared matrix builders for Q4_K and Q5_K tensors with optional group mins.
 * @author bhangun
 */
final class GgufQ45KBuild {
    private static final float[] EMPTY_FLOAT_ARRAY = new float[0];
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ45KBuild() {
    }

    static Q4KMatrix q4(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q4_K.id) {
            throw new IllegalArgumentException("Tensor is not Q4_K: " + tensor.name());
        }
        K32Build build = build(model, tensor);
        Q4KMatrix matrix = new Q4KMatrix(
                build.columns(),
                build.rows(),
                build.blocksPerRow(),
                build.quants(),
                build.groupScales(),
                build.hasGroupMins() ? build.groupMins() : EMPTY_FLOAT_ARRAY,
                build.hasGroupMins());
        return remember(model, tensor, matrix);
    }

    static Q5KMatrix q5(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        if (tensor.typeId() != GgmlType.Q5_K.id) {
            throw new IllegalArgumentException("Tensor is not Q5_K: " + tensor.name());
        }
        K32Build build = build(model, tensor);
        Q5KMatrix matrix = new Q5KMatrix(
                build.columns(),
                build.rows(),
                build.blocksPerRow(),
                build.quants(),
                build.groupScales(),
                build.hasGroupMins() ? build.groupMins() : EMPTY_FLOAT_ARRAY,
                build.hasGroupMins());
        return remember(model, tensor, matrix);
    }

    private static K32Build build(GGUFModel model, GGUFTensorInfo tensor) {
        int columns = checkedColumns(tensor, Integer.MAX_VALUE);
        int rows = checkedRows(tensor);
        int blocksPerRow = columns / QK_K;
        int totalBlocks = Math.multiplyExact(rows, blocksPerRow);
        MemorySegment source = tensorData(model, tensor);
        byte[] quants = new byte[Math.multiplyExact(totalBlocks, QK_K)];
        float[] groupScales = new float[Math.multiplyExact(totalBlocks, 8)];
        GgufKScales.State minState = new GgufKScales.State();
        boolean mayHaveGroupMins = cachedKHasMins(model, tensor, totalBlocks, 8) != Boolean.FALSE;

        if (tensor.typeId() == GgmlType.Q4_K.id) {
            fillQ4Blocks(source, totalBlocks, groupScales, minState, mayHaveGroupMins, quants);
        } else {
            fillQ5Blocks(source, totalBlocks, groupScales, minState, mayHaveGroupMins, quants);
        }
        return new K32Build(
                columns,
                rows,
                blocksPerRow,
                quants,
                groupScales,
                minState.groupMins,
                minState.hasGroupMins);
    }

    private static void fillQ4Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] groupScales,
            GgufKScales.State minState,
            boolean mayHaveGroupMins,
            byte[] quants) {
        long blockOffset = 0L;
        int groupBase = 0;
        int unpackedBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            fillQ4Block(source, blockOffset, groupScales, minState, mayHaveGroupMins, groupBase, quants, unpackedBase);
            blockOffset += Q4_K_BLOCK_BYTES;
            groupBase += 8;
            unpackedBase += QK_K;
        }
    }

    private static void fillQ5Blocks(
            MemorySegment source,
            int totalBlocks,
            float[] groupScales,
            GgufKScales.State minState,
            boolean mayHaveGroupMins,
            byte[] quants) {
        long blockOffset = 0L;
        int groupBase = 0;
        int unpackedBase = 0;
        for (int block = 0; block < totalBlocks; block++) {
            fillQ5Block(source, blockOffset, groupScales, minState, mayHaveGroupMins, groupBase, quants, unpackedBase);
            blockOffset += Q5_K_BLOCK_BYTES;
            groupBase += 8;
            unpackedBase += QK_K;
        }
    }

    private static void fillQ4Block(
            MemorySegment source,
            long blockOffset,
            float[] groupScales,
            GgufKScales.State minState,
            boolean mayHaveGroupMins,
            int groupBase,
            byte[] quants,
            int unpackedBase) {
        float d = f16ToF32(source.get(LE_SHORT, blockOffset));
        float dMin = mayHaveGroupMins ? f16ToF32(source.get(LE_SHORT, blockOffset + 2)) : 0.0f;
        long scalesOffset = blockOffset + 4;
        long scalesLow = source.get(LE_LONG, scalesOffset);
        int scalesHigh = source.get(LE_INT, scalesOffset + Long.BYTES);
        GgufKScales.fill(
                scalesLow,
                scalesHigh,
                d,
                dMin,
                mayHaveGroupMins,
                groupScales,
                minState,
                groupBase);

        long packedBase = blockOffset + 16;
        int outBase = unpackedBase;
        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            unpackQ4KPreparedGroup(source, packedBase, quants, outBase);
            outBase += 64;
            packedBase += 32;
        }
    }

    private static void fillQ5Block(
            MemorySegment source,
            long blockOffset,
            float[] groupScales,
            GgufKScales.State minState,
            boolean mayHaveGroupMins,
            int groupBase,
            byte[] quants,
            int unpackedBase) {
        float d = f16ToF32(source.get(LE_SHORT, blockOffset));
        float dMin = mayHaveGroupMins ? f16ToF32(source.get(LE_SHORT, blockOffset + 2)) : 0.0f;
        long scalesOffset = blockOffset + 4;
        long scalesLow = source.get(LE_LONG, scalesOffset);
        int scalesHigh = source.get(LE_INT, scalesOffset + Long.BYTES);
        GgufKScales.fill(
                scalesLow,
                scalesHigh,
                d,
                dMin,
                mayHaveGroupMins,
                groupScales,
                minState,
                groupBase);

        long packedBase = blockOffset + 48;
        int outBase = unpackedBase;
        int highMaskLow = 1;
        int highMaskHigh = 2;
        int highShiftLow = 0;
        int highShiftHigh = 1;
        for (int superBlockOffset = 0; superBlockOffset < QK_K; superBlockOffset += 64) {
            unpackQ5KPreparedGroup(
                    source,
                    packedBase,
                    blockOffset + 16,
                    highMaskLow,
                    highMaskHigh,
                    highShiftLow,
                    highShiftHigh,
                    quants,
                    outBase);
            highMaskLow <<= 2;
            highMaskHigh <<= 2;
            highShiftLow += 2;
            highShiftHigh += 2;
            outBase += 64;
            packedBase += 32;
        }
    }

    private record K32Build(
            int columns,
            int rows,
            int blocksPerRow,
            byte[] quants,
            float[] groupScales,
            float[] groupMins,
            boolean hasGroupMins) {}
}
