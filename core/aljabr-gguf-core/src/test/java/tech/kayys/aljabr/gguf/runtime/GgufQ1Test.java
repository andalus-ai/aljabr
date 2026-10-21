package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q1_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q1_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ1_0Block;

class GgufQ1Test {
    @Test
    void supportsQ1_0RowDotAndMatVec() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ1_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0xFF);
            writeQ1_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q1_0", new long[]{128, 2}, 41, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(41));

            float[] row = new float[128];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            GgufTensorOps.dequantizeRow(model, tensor, 1, row);
            for (float value : row) {
                assertEquals(-1.0f, value, 0.0f);
            }

            float[] vector = ones(128);
            assertEquals(128.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(-128.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(-128.0f, output[1], 0.0f);

            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(264L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(128, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(128.0f, preparedOutput[0], 0.0f);
            assertEquals(-128.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void rawQ1_0FastPathHandlesUnrolledBlocksAndTail() {
        int blocks = 5;
        int columns = blocks * Q1_0_BLOCK_SIZE;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(blocks * (long) Q1_0_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeQ1_0Block(
                        segment.asSlice(block * (long) Q1_0_BLOCK_BYTES, Q1_0_BLOCK_BYTES),
                        (short) 0x3c00,
                        (byte) (0x33 + block));
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q1_0.raw.unrolled_tail",
                    new long[]{columns, 1},
                    41,
                    0,
                    blocks * (long) Q1_0_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 29 - 14) * 0.03125f;
            }
            float[] row = dequantizedRow(model, tensor);

            assertEquals(
                    dot(row, vector),
                    GgufTqRawDot.dotRowQ1_0(segment, 0, columns, vector, 0),
                    1.0e-4f);
        }
    }
}
