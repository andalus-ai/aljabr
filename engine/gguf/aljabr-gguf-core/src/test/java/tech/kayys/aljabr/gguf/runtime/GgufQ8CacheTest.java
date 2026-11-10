package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;

class GgufQ8CacheTest {
    @Test
    void boundsPreparedQ8MatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "36");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 34);
            writeQ8Block(segment.asSlice(0, 34), (short) 0x3c00, (byte) 1);
            writeQ8Block(segment.asSlice(34, 34), (short) 0x3c00, (byte) 2);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q8.first", new long[]{32, 1}, 8, 0, 34);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q8.second", new long[]{32, 1}, 8, 34, 34);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q8Matrix first = GgufTensorOps.q8MatrixCached(model, firstTensor);
            assertEquals(36L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));
            assertEquals(36L, GgufTensorOps.q8MatrixCacheBytes(model));

            GgufTensorOps.q8MatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q8MatrixCacheSize(model));

            GgufTensorOps.Q8Matrix firstAfterEviction = GgufTensorOps.q8MatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ8MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previous);
        }
    }
}
