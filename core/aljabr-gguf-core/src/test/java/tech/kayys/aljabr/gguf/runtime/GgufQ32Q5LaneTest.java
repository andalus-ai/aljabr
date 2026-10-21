package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.dequantizedRow;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.ramp;
import static tech.kayys.aljabr.gguf.runtime.GgufFx.restoreProperty;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.expectedQ5SmallLaneOrderRow;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.expectedQ5SmallPreparedQuants;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_0LaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1LaneOrderBlock;

class GgufQ32Q5LaneTest {
    @Test
    void rawQ5MatVecPreservesHighBitNibbleLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            int highBits = 0xFFFF0000;
            MemorySegment q5_0Segment = arena.allocate(22);
            writeQ5_0LaneOrderBlock(q5_0Segment, (short) 0x3c00, highBits);
            GGUFTensorInfo q5_0Tensor = new GGUFTensorInfo("q5_0.raw.lanes", new long[]{32, 1}, 6, 0, 22);
            GGUFModel q5_0Model = new GGUFModel(3, Map.of(), List.of(q5_0Tensor), 0, q5_0Segment, null);

            float[] q5_0Output = new float[1];
            GgufTensorOps.matVecRows(q5_0Model, q5_0Tensor, ramp(32), q5_0Output, 1, true);
            assertEquals(1784.0f, q5_0Output[0], 0.0f);
            assertArrayEquals(expectedQ5SmallLaneOrderRow(highBits, 16, 0.0f),
                    dequantizedRow(q5_0Model, q5_0Tensor), 0.0f);
            assertArrayEquals(expectedQ5SmallPreparedQuants(highBits, 16),
                    GgufTensorOps.q32Matrix(q5_0Model, q5_0Tensor).quants());
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(q5_0Model));

            MemorySegment q5_1Segment = arena.allocate(24);
            writeQ5_1LaneOrderBlock(q5_1Segment, (short) 0x3c00, (short) 0, highBits);
            GGUFTensorInfo q5_1Tensor = new GGUFTensorInfo("q5_1.raw.lanes", new long[]{32, 1}, 7, 0, 24);
            GGUFModel q5_1Model = new GGUFModel(3, Map.of(), List.of(q5_1Tensor), 0, q5_1Segment, null);

            float[] q5_1Output = new float[1];
            GgufTensorOps.matVecRows(q5_1Model, q5_1Tensor, ramp(32), q5_1Output, 1, true);
            assertEquals(10232.0f, q5_1Output[0], 0.0f);
            assertArrayEquals(expectedQ5SmallLaneOrderRow(highBits, 0, 0.0f),
                    dequantizedRow(q5_1Model, q5_1Tensor), 0.0f);
            assertArrayEquals(expectedQ5SmallPreparedQuants(highBits, 0),
                    GgufTensorOps.q32Matrix(q5_1Model, q5_1Tensor).quants());
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(q5_1Model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousMaxBytes);
        }
    }
}
