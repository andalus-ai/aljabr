package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw Q32-family metadata probes for affine block-bias routing.
 * @author bhangun
 */
final class GgufQ32Meta {
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufQ32Meta() {
    }

    static boolean q4_1RowsHaveBlockBiases(MemorySegment source, long rowBytes, int columns, int rowCount) {
        return q32RowsHaveBlockBiases(source, rowBytes, columns, rowCount, Q4_1_BLOCK_BYTES);
    }

    static boolean q4_1RowHasBlockBiases(MemorySegment source, long rowOffset, int columns) {
        return q32RowHasBlockBiases(source, rowOffset, columns, Q4_1_BLOCK_BYTES);
    }

    static boolean q4_1BlocksHaveBiases(MemorySegment source, int totalBlocks) {
        return q32BlocksHaveBiases(source, totalBlocks, Q4_1_BLOCK_BYTES);
    }

    static boolean q5_1RowsHaveBlockBiases(MemorySegment source, long rowBytes, int columns, int rowCount) {
        return q32RowsHaveBlockBiases(source, rowBytes, columns, rowCount, Q5_1_BLOCK_BYTES);
    }

    static boolean q5_1RowHasBlockBiases(MemorySegment source, long rowOffset, int columns) {
        return q32RowHasBlockBiases(source, rowOffset, columns, Q5_1_BLOCK_BYTES);
    }

    static boolean q5_1BlocksHaveBiases(MemorySegment source, int totalBlocks) {
        return q32BlocksHaveBiases(source, totalBlocks, Q5_1_BLOCK_BYTES);
    }

    private static boolean q32RowsHaveBlockBiases(
            MemorySegment source,
            long rowBytes,
            int columns,
            int rowCount,
            int blockBytes) {
        int blocksPerRow = columns / Q4_0_BLOCK_SIZE;
        if (rowBytes == blocksPerRow * (long) blockBytes) {
            return q32BlocksHaveBiases(source, rowCount * blocksPerRow, blockBytes);
        }
        for (int row = 0; row < rowCount; row++) {
            if (q32RowHasBlockBiasesBlocks(source, row * rowBytes, blockBytes, blocksPerRow)) {
                return true;
            }
        }
        return false;
    }

    private static boolean q32RowHasBlockBiases(MemorySegment source, long rowOffset, int columns, int blockBytes) {
        int blocksPerRow = columns / Q4_0_BLOCK_SIZE;
        return q32RowHasBlockBiasesBlocks(source, rowOffset, blockBytes, blocksPerRow);
    }

    private static boolean q32RowHasBlockBiasesBlocks(
            MemorySegment source,
            long rowOffset,
            int blockBytes,
            int blocksPerRow) {
        for (int block = 0; block < blocksPerRow; block++) {
            if (q32BlockHasBias(source, rowOffset + block * (long) blockBytes)) {
                return true;
            }
        }
        return false;
    }

    private static boolean q32BlocksHaveBiases(MemorySegment source, int totalBlocks, int blockBytes) {
        int block = 0;
        int unrolledLimit = totalBlocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            long blockOffset = block * (long) blockBytes;
            if (q32BlockHasBias(source, blockOffset)
                    || q32BlockHasBias(source, blockOffset + blockBytes)
                    || q32BlockHasBias(source, blockOffset + 2L * blockBytes)
                    || q32BlockHasBias(source, blockOffset + 3L * blockBytes)) {
                return true;
            }
        }
        for (; block < totalBlocks; block++) {
            if (q32BlockHasBias(source, block * (long) blockBytes)) {
                return true;
            }
        }
        return false;
    }

    private static boolean q32BlockHasBias(MemorySegment source, long blockOffset) {
        return f16ToF32(source.get(LE_SHORT, blockOffset + Short.BYTES)) != 0.0f;
    }
}
