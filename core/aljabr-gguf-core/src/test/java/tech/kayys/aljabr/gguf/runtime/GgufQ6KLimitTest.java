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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ6KBlock;

class GgufQ6KLimitTest {
    @Test
    void boundsPreparedQ6KMatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("aljabr.gguf.q6k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q6k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 210);
            writeQ6KBlock(segment.asSlice(0, 210), (byte) 0x11, (byte) 0xAA, (byte) 1);
            writeQ6KBlock(segment.asSlice(210, 210), (byte) 0x22, (byte) 0xAA, (byte) 1);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q6.first", new long[]{256, 1}, 14, 0, 210);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q6.second", new long[]{256, 1}, 14, 210, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q6KMatrix first = GgufTensorOps.q6KMatrixCached(model, firstTensor);
            assertEquals(320L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q6KMatrixCacheSize(model));
            assertEquals(320L, GgufTensorOps.q6KMatrixCacheBytes(model));

            GgufTensorOps.q6KMatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q6KMatrixCacheSize(model));
            assertTrue(GgufTensorOps.q6KMatrixCacheBytes(model) <= 320L);

            GgufTensorOps.Q6KMatrix firstAfterEviction = GgufTensorOps.q6KMatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ6KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q6k.cache_max_bytes", previous);
        }
    }
}
