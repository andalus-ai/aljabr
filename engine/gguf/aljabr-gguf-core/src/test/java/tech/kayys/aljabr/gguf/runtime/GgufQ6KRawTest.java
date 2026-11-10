package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q6_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ6KRawTest {
    @Test
    void rawQ6KMatVecStreamsWhenPreparedCacheIsTooSmall() {
        String previousMinRows = System.getProperty("aljabr.gguf.q6k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q6k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q6k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q6k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KBlock(segment.asSlice(0, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ6KBlock(segment.asSlice(210, 210), (byte) 0x22, (byte) 0xAA, (byte) 1);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6.raw.cache_miss", new long[]{256, 2}, 14, 0, 2L * 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q6KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q6k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q6k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ6KMatVecWithGroupSumsPreservesLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q6k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q6k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q6k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q6k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KLaneOrderBlock(segment.asSlice(0, 210));
            writeQ6KLaneOrderBlock(segment.asSlice(210, 210));
            GGUFTensorInfo tensor = new GGUFTensorInfo("q6.raw.group_sums", new long[]{256, 2}, 14, 0, 2L * 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = ramp(256);
            float[] groupSums = GgufSum.vector16GroupSums(vector, 256, new GgufTensorOps.Q4KWorkBuffer());
            float expected = GgufQ6LaneFx.expectedDot(vector);
            assertEquals(expected, GgufQ6RawDot.dotRowQ6KWithGroupSums(segment, 0, 256, vector, groupSums), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);

            assertEquals(expected, output[0], 0.0f);
            assertEquals(expected, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q6KMatrixCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q6k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q6k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ6KGroupSumFastPathHandlesUnrolledBlocksAndTail() {
        int blocks = 5;
        int columns = blocks * QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(blocks * (long) Q6_K_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ6KBlock(
                        segment.asSlice(block * (long) Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES),
                        (byte) (0x11 + block),
                        (byte) (0xAA - block),
                        (byte) (1 + block));
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 43 - 21) * 0.01953125f;
            }
            float[] groupSums = GgufSum.vector16GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());

            assertEquals(
                    GgufQ6RawDot.dotRowQ6K(segment, 0, columns, vector, 0),
                    GgufQ6RawDot.dotRowQ6KWithGroupSums(segment, 0, columns, vector, groupSums),
                    1.0e-3f);
        }
    }

    @Test
    void rawQ6KRowWalkerHandlesUnrolledRowsAndBlockTail() {
        int rows = 5;
        int blocks = 5;
        int columns = blocks * QK_K;
        long rowBytes = blocks * (long) Q6_K_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < blocks; block++) {
                    writeQ6KBlock(
                            segment.asSlice(rowOffset + block * (long) Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES),
                            (byte) (0x11 + row * 3 + block),
                            (byte) (0xAA - row - block),
                            (byte) (1 + row + block));
                }
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 47 - 23) * 0.015625f;
            }
            float[] groupSums = GgufSum.vector16GroupSums(vector, columns, new GgufTensorOps.Q4KWorkBuffer());
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            for (int row = 0; row < rows; row++) {
                expected[row] = GgufQ6RawDot.dotRowQ6KWithGroupSums(
                        segment, row * rowBytes, columns, vector, groupSums);
            }

            GgufKRawRows.fillMatVecRowsQ6K(segment, columns, rowBytes, vector, groupSums, actual, 0, rows);

            assertArrayEquals(expected, actual, 1.0e-3f);
        }
    }
}
