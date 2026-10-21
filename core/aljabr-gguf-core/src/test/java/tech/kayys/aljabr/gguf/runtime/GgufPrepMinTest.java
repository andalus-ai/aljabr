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
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;

class GgufPrepMinTest {
    @Test
    void prepareMatrixCachesHonorsMinimumRows() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(86);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ8Block(segment.asSlice(18, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(52, 34), (short) 0x3c00, (byte) 2);

            GGUFTensorInfo small = new GGUFTensorInfo("small.q4_0", new long[]{32, 1}, 2, 0, 18);
            GGUFTensorInfo large = new GGUFTensorInfo("large.q8", new long[]{32, 2}, 8, 18, 68);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(small, large), 0, segment, null);

            GgufTensorOps.PreparedMatrixCacheStats stats =
                    GgufTensorOps.prepareMatrixCaches(model, model.tensors(), 2);

            assertTrue(stats.ready());
            assertEquals(2, stats.scannedTensors());
            assertEquals(2, stats.matrixTensors());
            assertEquals(1, stats.preparedCandidates());
            assertEquals(1, stats.preparedTensors());
            assertEquals(0, stats.skippedUnsupportedTypeTensors());
            assertEquals(1, stats.skippedSmallRowTensors());
            assertEquals(0, stats.skippedCacheTooSmallTensors());
            assertEquals(0, stats.failedTensors());
            assertEquals(72L, stats.preparedBytes());
            assertEquals(1, stats.cacheEntries());
            assertEquals(72L, stats.cacheBytes());
            assertEquals(1, GgufTensorOps.clearPreparedMatrixCaches(model));
        }
    }
}
