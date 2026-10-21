package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ2_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ2_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufTQFx.assertTQ2Pattern;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufTQ2Test {
    @Test
    void supportsTQ2_0RowDotAndPreparedMatVec() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 66);
            writeTQ2_0Block(segment.asSlice(0, 66), (short) 0x3c00, (byte) 0x24);
            writeTQ2_0Block(segment.asSlice(66, 66), (short) 0x3c00, (byte) 0xAA);
            GGUFTensorInfo tensor = new GGUFTensorInfo("tq2_0", new long[]{256, 2}, 35, 0, 2L * 66);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(35));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            assertTQ2Pattern(row, -1.0f, 0.0f, 1.0f, -1.0f);

            GgufTensorOps.dequantizeRow(model, tensor, 1, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(-64.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(-64.0f, output[0], 0.0f);
            assertEquals(256.0f, output[1], 0.0f);

            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(520L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(-64.0f, preparedOutput[0], 0.0f);
            assertEquals(256.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void rawTQ2_0FastPathHandlesUnrolledBlocksAndTail() {
        int blocks = 5;
        int columns = blocks * TQ2_0_BLOCK_SIZE;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(blocks * (long) TQ2_0_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeTQ2_0Block(
                        segment.asSlice(block * (long) TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES),
                        (short) 0x3c00,
                        (byte) (0x24 + block));
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "tq2_0.raw.unrolled_tail",
                    new long[]{columns, 1},
                    35,
                    0,
                    blocks * (long) TQ2_0_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 37 - 18) * 0.0234375f;
            }
            float[] row = dequantizedRow(model, tensor);

            assertEquals(
                    dot(row, vector),
                    GgufTqRawDot.dotRowTQ2_0(segment, 0, columns, vector, 0),
                    1.0e-3f);
        }
    }
}
