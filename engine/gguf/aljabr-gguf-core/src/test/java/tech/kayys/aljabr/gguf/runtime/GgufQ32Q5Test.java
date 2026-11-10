package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;

class GgufQ32Q5Test {
    @Test
    void supportsQ5_0RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 22);
            writeQ5_0Block(segment.asSlice(0, 22), (short) 0x3c00, -1, (byte) 0x10);
            writeQ5_0Block(segment.asSlice(22, 22), (short) 0x3c00, -1, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_0", new long[]{32, 2}, 6, 0, 2L * 22);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(6));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(0.0f, row[i], 0.0f);
                assertEquals(1.0f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(16.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);
            assertEquals(48.0f, GgufTensorOps.dotRow(model, tensor, 1, vector), 0.0f);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 2, true);
            assertEquals(16.0f, output[0], 0.0f);
            assertEquals(48.0f, output[1], 0.0f);
        }
    }

    @Test
    void supportsQ5_1RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(24);
            writeQ5_1Block(segment, (short) 0x3c00, (short) 0x3800, -1, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5_1", new long[]{32, 1}, 7, 0, 24);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            assertTrue(GgufTensorOps.supportsRowDotType(7));

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(17.5f, row[i], 0.0f);
                assertEquals(18.5f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(576.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(576.0f, output[0], 0.0f);
        }
    }
}
