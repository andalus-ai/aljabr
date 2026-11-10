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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;

class GgufQ5KCacheTest {
    @Test
    void boundsPreparedQ5KMatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            writeQ5KBlock(segment.asSlice(176, 176), (byte) 0xFF, (byte) 0x11);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q5.first", new long[]{256, 1}, 13, 0, 176);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q5.second", new long[]{256, 1}, 13, 176, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q5KMatrix first = GgufTensorOps.q5KMatrixCached(model, firstTensor);
            assertEquals(288L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(288L, GgufTensorOps.q5KMatrixCacheBytes(model));

            GgufTensorOps.q5KMatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertTrue(GgufTensorOps.q5KMatrixCacheBytes(model) <= 320L);

            GgufTensorOps.Q5KMatrix firstAfterEviction = GgufTensorOps.q5KMatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ5KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previous);
        }
    }
}
