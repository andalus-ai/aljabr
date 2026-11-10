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
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlockWithMin;

class GgufQ2KMixTest {
    @Test
    void rawQ2KMatVecSkipsMinCorrectionForZeroEncodedMinBlocksInMixedTensor() {
        String previousMinRows = System.getProperty("aljabr.gguf.q2k.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q2k.cache_max_bytes");
        System.setProperty("aljabr.gguf.q2k.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q2k.cache_max_bytes", "320");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 84);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            segment.set(LE_SHORT, 82, (short) 0x7e00);
            writeQ2KBlockWithMin(segment.asSlice(84, 84), (byte) 0x13, (byte) 0x55);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q2.raw.mixed_zero_dmin", new long[]{256, 2}, 10, 0, 2L * 84);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            float[] output = new float[2];
            GgufTensorOps.matVecRows(model, tensor, ones(256), output, 2, true);

            assertEquals(256.0f, output[0], 0.0f);
            assertEquals(512.0f, output[1], 0.0f);
            assertEquals(0, GgufTensorOps.q2KMatrixCacheSize(model));
            assertEquals(768L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, tensor));
        } finally {
            restoreProperty("aljabr.gguf.q2k.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q2k.cache_max_bytes", previousMaxBytes);
        }
    }
}
