package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ones;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.LE_SHORT;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlockWithMin;

class GgufQ5KMixTest {
    @Test
    void rawQ5KMatVecSkipsMinCorrectionForZeroEncodedMinBlocksInMixedTensor() {
        String previousMinRows = System.getProperty("aljabr.gguf.q5k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q5k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q5k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q5k.cache_max_bytes", "288");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 176);
            writeQ5KBlock(segment.asSlice(0, 176), (byte) 0xFF, (byte) 0);
            segment.set(LE_SHORT, 2, (short) 0x7e00);
            writeQ5KBlockWithMin(segment.asSlice(176, 176), (byte) 2, (byte) 1, (byte) 0, (byte) 0x11);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q5.raw.mixed_zero_dmin", new long[]{256, 2}, 13, 0, 2L * 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 2, true);

            assertEquals(4096.0f, output[0], 0.0f);
            assertEquals(256.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q5KMatrixCacheSize(model));
            assertEquals(640L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
        } finally {
            restoreProperty("aljabr.gguf.q5k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q5k.cache_max_bytes", previousMaxBytes);
        }
    }
}
