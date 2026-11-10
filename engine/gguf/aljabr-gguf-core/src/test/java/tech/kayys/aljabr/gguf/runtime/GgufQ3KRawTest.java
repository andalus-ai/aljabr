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
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q3_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ3KRawTest {
    @Test
    void rawQ3KMatVecStreamsWhenPreparedCacheIsTooSmall() {
        String previousMinRows = System.getProperty("aljabr.gguf.q3k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q3k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q3k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q3k.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 110);
            writeQ3KBlock(segment.asSlice(0, 110), 1, (byte) 0x55, (byte) 0xFF);
            writeQ3KBlock(segment.asSlice(110, 110), 1, (byte) 0xAA, (byte) 0xFF);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q3.raw.cache_miss", new long[]{256, 2}, 11, 0, 2L * 110);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q3KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
            assertEquals(1, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
        } finally {
            restoreProperty("aljabr.gguf.q3k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q3k.cache_max_bytes", previousMaxBytes);
        }
    }

    @Test
    void rawQ3KFastPathHandlesUnrolledBlocksAndTail() {
        int blocks = 5;
        int columns = blocks * QK_K;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(blocks * (long) Q3_K_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ3KBlock(
                        segment.asSlice(block * (long) Q3_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES),
                        1 + block,
                        (byte) (0x55 ^ block),
                        (byte) (0xF0 - block));
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q3.raw.unrolled_tail",
                    new long[]{columns, 1},
                    11,
                    0,
                    blocks * (long) Q3_K_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 41 - 20) * 0.0234375f;
            }
            float[] row = dequantizedRow(model, tensor);

            assertEquals(
                    dot(row, vector),
                    GgufQ3RawDot.dotRowQ3K(segment, 0, columns, vector, 0),
                    1.0e-3f);
        }
    }

    @Test
    void rawQ3KRowWalkerHandlesUnrolledRowsAndBlockTail() {
        int rows = 5;
        int blocks = 5;
        int columns = blocks * QK_K;
        long rowBytes = blocks * (long) Q3_K_BLOCK_BYTES;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(rows * rowBytes);
            for (int row = 0; row < rows; row++) {
                long rowOffset = row * rowBytes;
                for (int block = 0; block < blocks; block++) {
                    writeQ3KBlock(
                            segment.asSlice(rowOffset + block * (long) Q3_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES),
                            1 + row + block,
                            (byte) (0x55 ^ row ^ block),
                            (byte) (0xF0 - row - block));
                }
            }

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 41 - 20) * 0.0234375f;
            }
            float[] expected = new float[rows];
            float[] actual = new float[rows];
            for (int row = 0; row < rows; row++) {
                expected[row] = GgufQ3RawDot.dotRowQ3K(segment, row * rowBytes, columns, vector, 0);
            }

            GgufKRawRows.fillMatVecRowsQ3K(segment, columns, rowBytes, vector, actual, 0, rows);

            assertArrayEquals(expected, actual, 1.0e-3f);
        }
    }

    private static float dot(float[] left, float[] right) {
        float sum = 0.0f;
        for (int index = 0; index < left.length; index++) {
            sum += left[index] * right[index];
        }
        return sum;
    }
}
