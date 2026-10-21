package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ32CacheTest {
    @Test
    void boundsPreparedQ32MatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "40");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 18);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4_0Block(segment.asSlice(18, 18), (short) 0x3c00, (byte) 0xA9);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4_0.first", new long[]{32, 1}, 2, 0, 18);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4_0.second", new long[]{32, 1}, 2, 18, 18);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q32Matrix first = GgufTensorOps.q32MatrixCached(model, firstTensor);
            assertEquals(36L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q32MatrixCacheSize(model));
            assertEquals(36L, GgufTensorOps.q32MatrixCacheBytes(model));

            GgufTensorOps.q32MatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q32MatrixCacheSize(model));
            assertTrue(GgufTensorOps.q32MatrixCacheBytes(model) <= 40L);

            GgufTensorOps.Q32Matrix firstAfterEviction = GgufTensorOps.q32MatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ32MatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previous);
        }
    }
}
