package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ4KCacheTest {
    @Test
    void boundsPreparedQ4KMatrixCacheByEstimatedBytes() {
        String previous = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, firstTensor);
            assertEquals(320L, first.estimatedBytes());
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(320L, GgufTensorOps.q4KMatrixCacheBytes(model));

            GgufTensorOps.q4KMatrixCached(model, secondTensor);
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));
            assertTrue(GgufTensorOps.q4KMatrixCacheBytes(model) <= 320L);

            GgufTensorOps.Q4KMatrix firstAfterEviction = GgufTensorOps.q4KMatrixCached(model, firstTensor);
            assertNotSame(first, firstAfterEviction);
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previous);
        }
    }

    @Test
    void canDisablePreparedQ4KMatrixCache() {
        String previous = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KBlockWithAllScalesAndMins(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, tensor);
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));

            System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "0");
            GgufTensorOps.Q4KMatrix second = GgufTensorOps.q4KMatrixCached(model, tensor);

            assertNotSame(first, second);
            assertEquals(0, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(0L, GgufTensorOps.q4KMatrixCacheBytes(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previous);
        }
    }

    @Test
    void evictsPreparedQ4KMatrixCacheWhenBudgetShrinksBelowCurrentBytes() {
        String previous = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "640");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.q4KMatrixCached(model, firstTensor);
            GgufTensorOps.q4KMatrixCached(model, secondTensor);
            assertEquals(2, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.q4KMatrixCacheBytes(model));

            System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "320");
            GgufTensorOps.q4KMatrixCached(model, secondTensor);

            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));
            assertEquals(320L, GgufTensorOps.q4KMatrixCacheBytes(model));
            assertEquals(1, GgufTensorOps.clearQ4KMatrixCache(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previous);
        }
    }
}
