package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;

import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.decoderCacheModel;

class GgufProbeSmallTest {
    @Test
    void autoSkipsDecoderCachesWhenPreparedMatricesCannotFitConfiguredCaches() {
        String previousQ32MaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        String previousQ8MaxBytes = System.getProperty("aljabr.gguf.q8.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "1");
        System.setProperty("aljabr.gguf.q8.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            GGUFModel model = decoderCacheModel(arena);
            GgufRuntimeProbe.PreparedMatrixCacheSelection selection =
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(model, 0, true, 1, 1024);

            assertEquals("auto-skipped-cache-too-small", selection.mode());
            assertFalse(selection.prepare());
            assertEquals(0, selection.plan().preparedCandidates());
            assertEquals(2, selection.plan().skippedCacheTooSmallTensors());
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousQ32MaxBytes);
            restoreProperty("aljabr.gguf.q8.cache_max_bytes", previousQ8MaxBytes);
        }
    }
}
