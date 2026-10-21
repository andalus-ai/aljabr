package tech.kayys.aljabr.gguf.runtime;

import static tech.kayys.aljabr.gguf.runtime.GgufQuantTables.f16ToF32;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q2_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Raw K-quant metadata probes used before choosing a dot path.
 *
 * <p>The raw Q2_K/Q4_K/Q5_K kernels have faster no-min and cached-vector-sum
 * paths. This helper scans only compact scale/min metadata so callers can pick
 * one of those paths once per mat-vec instead of rediscovering min state inside
 * every row dot.</p>
 * @author bhangun
 */
final class GgufKMeta {
    private static final long HIGH_NIBBLE_MASK = 0xF0F0F0F0F0F0F0F0L;
    private static final long K4_LOW_MIN_MASK = 0xFFFFFFFF00000000L;
    private static final int K4_HIGH_MIN_MASK = 0xF0F0F0F0;
    private static final ValueLayout.OfShort LE_SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt LE_INT = ValueLayout.JAVA_INT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgufKMeta() {
    }

    static boolean q2RowsHaveMins(MemorySegment source, long rowBytes, int columns, int rowCount) {
        int blocksPerRow = columns / QK_K;
        if (rowBytes == blocksPerRow * (long) Q2_K_BLOCK_BYTES) {
            return q2BlocksHaveMins(source, rowCount * blocksPerRow);
        }
        for (int row = 0; row < rowCount; row++) {
            if (q2RowHasMinsBlocks(source, row * rowBytes, blocksPerRow)) {
                return true;
            }
        }
        return false;
    }

    static boolean q2RowHasMins(MemorySegment source, long rowOffset, int columns) {
        int blocksPerRow = columns / QK_K;
        return q2RowHasMinsBlocks(source, rowOffset, blocksPerRow);
    }

    private static boolean q2RowHasMinsBlocks(MemorySegment source, long rowOffset, int blocksPerRow) {
        for (int block = 0; block < blocksPerRow; block++) {
            if (q2BlockHasMins(source, rowOffset + block * (long) Q2_K_BLOCK_BYTES)) {
                return true;
            }
        }
        return false;
    }

    static boolean q2BlocksHaveMins(MemorySegment source, int totalBlocks) {
        int block = 0;
        int unrolledLimit = totalBlocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            long blockOffset = block * (long) Q2_K_BLOCK_BYTES;
            if (q2BlockHasMins(source, blockOffset)
                    || q2BlockHasMins(source, blockOffset + Q2_K_BLOCK_BYTES)
                    || q2BlockHasMins(source, blockOffset + 2L * Q2_K_BLOCK_BYTES)
                    || q2BlockHasMins(source, blockOffset + 3L * Q2_K_BLOCK_BYTES)) {
                return true;
            }
        }
        for (; block < totalBlocks; block++) {
            if (q2BlockHasMins(source, block * (long) Q2_K_BLOCK_BYTES)) {
                return true;
            }
        }
        return false;
    }

    static boolean q4RowsHaveMins(MemorySegment source, long rowBytes, int columns, int rowCount) {
        return q4OrQ5RowsHaveMins(source, rowBytes, columns, rowCount, Q4_K_BLOCK_BYTES);
    }

    static boolean q4RowHasMins(MemorySegment source, long rowOffset, int columns) {
        return q4OrQ5RowHasMins(source, rowOffset, columns, Q4_K_BLOCK_BYTES);
    }

    static boolean q4BlocksHaveMins(MemorySegment source, int totalBlocks) {
        return q4OrQ5BlocksHaveMins(source, totalBlocks, Q4_K_BLOCK_BYTES);
    }

    static boolean q5RowsHaveMins(MemorySegment source, long rowBytes, int columns, int rowCount) {
        return q4OrQ5RowsHaveMins(source, rowBytes, columns, rowCount, Q5_K_BLOCK_BYTES);
    }

    static boolean q5RowHasMins(MemorySegment source, long rowOffset, int columns) {
        return q4OrQ5RowHasMins(source, rowOffset, columns, Q5_K_BLOCK_BYTES);
    }

    static boolean q5BlocksHaveMins(MemorySegment source, int totalBlocks) {
        return q4OrQ5BlocksHaveMins(source, totalBlocks, Q5_K_BLOCK_BYTES);
    }

    private static boolean q2BlockHasMins(MemorySegment source, long blockOffset) {
        float dMin = f16ToF32(source.get(LE_SHORT, blockOffset + 82));
        if (dMin == 0.0f) {
            return false;
        }
        long scaleMinsLow = source.get(LE_LONG, blockOffset);
        long scaleMinsHigh = source.get(LE_LONG, blockOffset + Long.BYTES);
        return ((scaleMinsLow | scaleMinsHigh) & HIGH_NIBBLE_MASK) != 0L;
    }

    private static boolean q4OrQ5RowsHaveMins(
            MemorySegment source,
            long rowBytes,
            int columns,
            int rowCount,
            int blockBytes) {
        int blocksPerRow = columns / QK_K;
        if (rowBytes == blocksPerRow * (long) blockBytes) {
            return q4OrQ5BlocksHaveMins(source, rowCount * blocksPerRow, blockBytes);
        }
        for (int row = 0; row < rowCount; row++) {
            if (q4OrQ5RowHasMinsBlocks(source, row * rowBytes, blockBytes, blocksPerRow)) {
                return true;
            }
        }
        return false;
    }

    private static boolean q4OrQ5RowHasMins(MemorySegment source, long rowOffset, int columns, int blockBytes) {
        int blocksPerRow = columns / QK_K;
        return q4OrQ5RowHasMinsBlocks(source, rowOffset, blockBytes, blocksPerRow);
    }

    private static boolean q4OrQ5RowHasMinsBlocks(
            MemorySegment source,
            long rowOffset,
            int blockBytes,
            int blocksPerRow) {
        for (int block = 0; block < blocksPerRow; block++) {
            if (q4OrQ5BlockHasMins(source, rowOffset + block * (long) blockBytes)) {
                return true;
            }
        }
        return false;
    }

    private static boolean q4OrQ5BlocksHaveMins(MemorySegment source, int totalBlocks, int blockBytes) {
        int block = 0;
        int unrolledLimit = totalBlocks - 4;
        for (; block <= unrolledLimit; block += 4) {
            long blockOffset = block * (long) blockBytes;
            if (q4OrQ5BlockHasMins(source, blockOffset)
                    || q4OrQ5BlockHasMins(source, blockOffset + blockBytes)
                    || q4OrQ5BlockHasMins(source, blockOffset + 2L * blockBytes)
                    || q4OrQ5BlockHasMins(source, blockOffset + 3L * blockBytes)) {
                return true;
            }
        }
        for (; block < totalBlocks; block++) {
            if (q4OrQ5BlockHasMins(source, block * (long) blockBytes)) {
                return true;
            }
        }
        return false;
    }

    private static boolean q4OrQ5BlockHasMins(MemorySegment source, long blockOffset) {
        float dMin = f16ToF32(source.get(LE_SHORT, blockOffset + 2));
        if (dMin == 0.0f) {
            return false;
        }
        long scalesOffset = blockOffset + 4;
        long scalesLow = source.get(LE_LONG, scalesOffset);
        int scalesHigh = source.get(LE_INT, scalesOffset + Long.BYTES);
        return (scalesLow & K4_LOW_MIN_MASK) != 0L || (scalesHigh & K4_HIGH_MIN_MASK) != 0;
    }
}
