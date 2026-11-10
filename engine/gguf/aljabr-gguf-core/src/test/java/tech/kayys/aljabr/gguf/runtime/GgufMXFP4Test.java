package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeMXFP4Block;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufMXFP4Test {
    @Test
    void supportsMXFP4RowDotAndPreparedMatVec() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_min_rows");
        System.setProperty("aljabr.gguf.q8.cache_min_rows", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 17);
            writeMXFP4Block(segment.asSlice(0, 17), (byte) 128, (byte) 0xA5);
            writeMXFP4Block(segment.asSlice(17, 17), (byte) 128, (byte) 0x3C);
            GGUFTensorInfo tensor = new GGUFTensorInfo("mxfp4", new long[]{32, 2}, 39, 0, 2L * 17);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(39));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(6.0f, row[i], 0.0f);
                assertEquals(-2.0f, row[16 + i], 0.0f);
            }

            GgufTensorOps.dequantizeRow(model, tensor, 1, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(-4.0f, row[i], 0.0f);
                assertEquals(3.0f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(64.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(-16.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(64.0f, output[0], 0.0f);
            assertEquals(-16.0f, output[1], 0.0f);

            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(72L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, tensor);
            GgufTensorOps.Q8Matrix second = GgufTensorOps.q8MatrixCached(model, tensor);
            assertSame(first, second);
            assertEquals(32, first.blockSize());

            float[] preparedOutput = new float[2];
            GgufTensorOps.matVecRows(first, vector, preparedOutput, 2, true);
            assertEquals(64.0f, preparedOutput[0], 0.0f);
            assertEquals(-16.0f, preparedOutput[1], 0.0f);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_min_rows", previous);
        }
    }
}
