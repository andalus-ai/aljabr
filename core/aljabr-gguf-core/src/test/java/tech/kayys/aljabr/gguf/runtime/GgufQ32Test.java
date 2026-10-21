package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.*;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ32Test {
    @Test
    void supportsQ4_0RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_0", new long[]{32, 2}, 2, 0, 2L * 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

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
    void supportsQ4_1RowDotAndMatVec() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(20);
            writeQ4_1Block(segment, (short) 0x3c00, (short) 0x3800, (byte) 0x21);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4_1", new long[]{32, 1}, 3, 0, 20);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] row = new float[32];
            GgufTensorOps.dequantizeRow(model, tensor, 0, row);
            for (int i = 0; i < 16; i++) {
                assertEquals(1.5f, row[i], 0.0f);
                assertEquals(2.5f, row[16 + i], 0.0f);
            }

            float[] vector = ones(32);
            assertEquals(64.0f, GgufTensorOps.dotRow(model, tensor, 0, vector), 0.0f);

            float[] output = new float[1];
            GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);
            assertEquals(64.0f, output[0], 0.0f);
        }
    }
}
