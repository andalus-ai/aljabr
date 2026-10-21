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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.LE_SHORT;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;

class GgufQ5KMissTest {
    @Test
    void genericMatVecStreamsNoMinQ5KCacheMissWithCachedNoMinEstimate() {
        String previousMinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "288");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            writeQ5KBlock(segment.asSlice(176, 176), (byte) 0xFF, (byte) 0x11);
            segment.set(LE_SHORT, 178, (short) 0x7e00);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q5.no_mins.first", new long[]{256, 1}, 13, 0, 176);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q5.no_mins.second", new long[]{256, 1}, 13, 176, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q5KMatrix first = GgufTensorOps.q5KMatrixCached(model, firstTensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(model, secondTensor, ones(256), output, 1, true);

            assertEquals(4352.0f, output[0], 0.0f);
            assertEquals(1, GgufTensorOps.q5KMatrixCacheSize(model));
            assertSame(first, GgufTensorOps.q5KMatrixCached(model, firstTensor));
            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(1, GgufTensorOps.clearPreparedMatrixCaches(model));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousMaxBytes);
        }
    }
}
