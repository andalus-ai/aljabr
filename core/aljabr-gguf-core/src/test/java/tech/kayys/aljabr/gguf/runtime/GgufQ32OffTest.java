package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ32OffTest {
    @Test
    void buildsPreparedQ32MatricesFromNonZeroTensorOffsets() {
        assertPreparedQ32Offset(
                2,
                18,
                (block, packedQuant) -> writeQ4_0Block(block, (short) 0x3c00, packedQuant),
                (byte) 0xA9,
                48.0f,
                36L);
        assertPreparedQ32Offset(
                3,
                20,
                (block, packedQuant) -> writeQ4_1Block(block, (short) 0x3c00, (short) 0x3800, packedQuant),
                (byte) 0x21,
                64.0f,
                40L);
        assertPreparedQ32Offset(
                6,
                22,
                (block, packedQuant) -> writeQ5_0Block(block, (short) 0x3c00, -1, packedQuant),
                (byte) 0x21,
                48.0f,
                36L);
        assertPreparedQ32Offset(
                7,
                24,
                (block, packedQuant) -> writeQ5_1Block(block, (short) 0x3c00, (short) 0x3800, -1, packedQuant),
                (byte) 0x21,
                576.0f,
                40L);
    }

    private static void assertPreparedQ32Offset(
            int typeId,
            int blockBytes,
            Q32BlockWriter writer,
            byte packedQuant,
            float expectedDot,
            long expectedBytes) {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * blockBytes);
            writer.write(segment.asSlice(0, blockBytes), (byte) 0);
            writer.write(segment.asSlice(blockBytes, blockBytes), packedQuant);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q32.offset." + typeId,
                    new long[]{32, 1},
                    typeId,
                    blockBytes,
                    blockBytes);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q32Matrix matrix = GgufTensorOps.q32Matrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(32), output, 1, true);

            assertEquals(expectedDot, output[0], 0.0f);
            assertEquals(expectedBytes, matrix.estimatedBytes());
        }
    }

    private interface Q32BlockWriter {
        void write(MemorySegment block, byte packedQuant);
    }
}
