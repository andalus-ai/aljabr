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
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.expectedQ4SmallLaneOrderRow;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.expectedQ4SmallPreparedQuants;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0LaneOrderBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1LaneOrderBlock;

class GgufQ32Q4LaneTest {
    @Test
    void rawQ4MatVecPreservesNibbleLaneOrder() {
        String previousMinRows = System.getProperty("aljabr.gguf.q32.cache_min_rows");
        String previousMaxBytes = System.getProperty("aljabr.gguf.q32.cache_max_bytes");
        System.setProperty("aljabr.gguf.q32.cache_min_rows", "1");
        System.setProperty("aljabr.gguf.q32.cache_max_bytes", "1");
        try (Arena arena = Arena.ofShared()) {
            MemorySegment q4_0Segment = arena.allocate(18);
            writeQ4_0LaneOrderBlock(q4_0Segment, (short) 0x3c00);
            GGUFTensorInfo q4_0Tensor = new GGUFTensorInfo("q4_0.raw.lanes", new long[]{32, 1}, 2, 0, 18);
            GGUFModel q4_0Model = new GGUFModel(3, Map.of(), List.of(q4_0Tensor), 0, q4_0Segment, null);

            float[] q4_0Output = new float[1];
            GgufTensorOps.matVecRows(q4_0Model, q4_0Tensor, ramp(32), q4_0Output, 1, true);
            assertEquals(-264.0f, q4_0Output[0], 0.0f);
            assertArrayEquals(expectedQ4SmallLaneOrderRow(8, 0.0f), dequantizedRow(q4_0Model, q4_0Tensor), 0.0f);
            assertArrayEquals(expectedQ4SmallPreparedQuants(8),
                    GgufTensorOps.q32Matrix(q4_0Model, q4_0Tensor).quants());
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(q4_0Model));

            MemorySegment q4_1Segment = arena.allocate(20);
            writeQ4_1LaneOrderBlock(q4_1Segment, (short) 0x3c00, (short) 0);
            GGUFTensorInfo q4_1Tensor = new GGUFTensorInfo("q4_1.raw.lanes", new long[]{32, 1}, 3, 0, 20);
            GGUFModel q4_1Model = new GGUFModel(3, Map.of(), List.of(q4_1Tensor), 0, q4_1Segment, null);

            float[] q4_1Output = new float[1];
            GgufTensorOps.matVecRows(q4_1Model, q4_1Tensor, ramp(32), q4_1Output, 1, true);
            assertEquals(3960.0f, q4_1Output[0], 0.0f);
            assertArrayEquals(expectedQ4SmallLaneOrderRow(0, 0.0f), dequantizedRow(q4_1Model, q4_1Tensor), 0.0f);
            assertArrayEquals(expectedQ4SmallPreparedQuants(0),
                    GgufTensorOps.q32Matrix(q4_1Model, q4_1Tensor).quants());
            assertEquals(0, GgufTensorOps.q32MatrixCacheSize(q4_1Model));
        } finally {
            restoreProperty("aljabr.gguf.q32.cache_min_rows", previousMinRows);
            restoreProperty("aljabr.gguf.q32.cache_max_bytes", previousMaxBytes);
        }
    }
}
