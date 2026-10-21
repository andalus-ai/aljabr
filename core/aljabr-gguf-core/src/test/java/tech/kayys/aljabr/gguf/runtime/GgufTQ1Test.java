package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.TQ1_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeTQ1_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufTQFx.assertTQ1Pattern;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufTQ1Test {
    @Test
    void supportsTQ1_0RowDotAndPreparedMatVec() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 54);
            writeTQ1_0Block(segment.asSlice(0, 54), (short) 0x3c00, (byte) 49);
            writeTQ1_0Block(segment.asSlice(54, 54), (short) 0x3c00, (byte) 0xFF);
            GGUFTensorInfo tensor = new GGUFTensorInfo("tq1_0", new long[]{256, 2}, 34, 0, 2L * 54);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(34));

            float[] row = new float[256];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            assertTQ1Pattern(row, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f);

            GgufTensorOps.dequantizeRow(model, tensor, 1, row);
            for (float value : row) {
                assertEquals(1.0f, value, 0.0f);
            }

            float[] vector = ones(256);
            assertEquals(-52.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(256.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(-52.0f, output[0], 0.0f);
            assertEquals(256.0f, output[1], 0.0f);

            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(520L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(256, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(-52.0f, preparedOutput[0], 0.0f);
            assertEquals(256.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previous);
        }
    }

    @Test
    void rawTQ1_0FastPathHandlesUnrolledBlocksAndTail() {
        int blocks = 5;
        int columns = blocks * TQ1_0_BLOCK_SIZE;
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(blocks * (long) TQ1_0_BLOCK_BYTES);
            for (int block = 0; block < blocks; block++) {
                writeTQ1_0Block(
                        segment.asSlice(block * (long) TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_BYTES),
                        (short) 0x3c00,
                        (byte) (49 + block));
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "tq1_0.raw.unrolled_tail",
                    new long[]{columns, 1},
                    34,
                    0,
                    blocks * (long) TQ1_0_BLOCK_BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] vector = new float[columns];
            for (int index = 0; index < vector.length; index++) {
                vector[index] = (index % 31 - 15) * 0.02734375f;
            }
            float[] row = dequantizedRow(model, tensor);

            assertEquals(
                    dot(row, vector),
                    GgufTqRawDot.dotRowTQ1_0(segment, 0, columns, vector, 0),
                    1.0e-3f);
        }
    }
}
