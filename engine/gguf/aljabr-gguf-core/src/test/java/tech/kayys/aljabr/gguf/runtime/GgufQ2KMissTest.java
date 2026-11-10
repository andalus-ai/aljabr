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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlock;

class GgufQ2KMissTest {
    @Test
    void genericMatVecStreamsNoMinQ2KCacheMissWithCachedNoMinEstimate() {
        String previousMinRows = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 84);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeQ2KBlock(segment.asSlice(84, 84), (byte) 0x01, (byte) 0xAA);
            segment.set(LE_SHORT, 166, (short) 0x7e00);
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q2.no_mins.first", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q2.no_mins.second", new long[]{256, 1}, 10, 84, 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            GgufTensorOps.Q2KMatrix first = GgufTensorOps.q2KMatrixCached(model, firstTensor);
            float[] output = new float[1];

            GgufTensorOps.matVecRows(model, secondTensor, ones(256), output, 1, true);

            assertEquals(512.0f, output[0], 0.0f);
            assertEquals(1, GgufTensorOps.q2KMatrixCacheSize(model));
            assertSame(first, GgufTensorOps.q2KMatrixCached(model, firstTensor));
            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(1, GgufTensorOps.clearPreparedMatrixCaches(model));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousMaxBytes);
        }
    }
}
