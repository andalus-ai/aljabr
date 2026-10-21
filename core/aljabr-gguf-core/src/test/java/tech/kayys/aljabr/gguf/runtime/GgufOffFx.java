package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;

/**
 * Shared non-zero tensor offset checks for prepared GGUF runtime matrices.
 */
final class GgufOffFx {
    private GgufOffFx() {
    }

    static void assertPreparedQ8Offset(
            int typeId,
            int columns,
            int blockBytes,
            Q8BlockWriter writer,
            byte packedQuant,
            float expectedDot,
            long expectedBytes) {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * blockBytes);
            writer.write(segment.asSlice(0, blockBytes), (byte) 0);
            writer.write(segment.asSlice(blockBytes, blockBytes), packedQuant);
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q8.offset." + typeId,
                    new long[]{columns, 1},
                    typeId,
                    blockBytes,
                    blockBytes);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q8Matrix matrix = GgufTensorOps.q8Matrix(model, tensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(matrix, ones(columns), output, 1, true);

            assertEquals(expectedDot, output[0], 0.0f);
            assertEquals(expectedBytes, matrix.estimatedBytes());
        }
    }

    interface Q8BlockWriter {
        void write(MemorySegment block, byte packedQuant);
    }
}
