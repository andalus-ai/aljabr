package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;

class GgufKMissTest {
    @Test
    void genericMatVecStreamsCacheMissWhenSharedCacheBucketIsFull() {
        String previousMinRows = System.getProperty("aljabr.gguf.q4k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q4k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q4k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q4k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q4KMatrix first = GgufTensorOps.q4KMatrixCached(model, firstTensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(model, secondTensor, ones(256), output, 1, true);

            assertEquals(128.0f, output[0], 0.0f);
            assertEquals(1, GgufTensorOps.q4KMatrixCacheSize(model));
            assertSame(first, GgufTensorOps.q4KMatrixCached(model, firstTensor));
            assertEquals(1, GgufTensorOps.clearPreparedMatrixCaches(model));
        } finally {
            restoreProperty("aljabr.gguf.q4k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q4k.cache_max_bytes", previousMaxBytes);
        }
    }
}
