package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufRuntimeProbeTest {
    private static final ValueLayout.OfFloat LE_FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void samplesPreferredTensorWithJavaRowDotPrimitive() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(6L * Float.BYTES);
            float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            for (int i = 0; i < values.length; i++) {
                segment.set(LE_FLOAT, i * (long) Float.BYTES, values[i]);
            }
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "blk.0.attn_q.weight",
                    new long[]{3, 2},
                    0,
                    0,
                    6L * Float.BYTES);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(model, 24, 2, 2);

            assertTrue(probe.hasTensorProbe());
            assertEquals("blk.0.attn_q.weight", probe.tensorName());
            assertEquals("F32", probe.tensorType());
            assertEquals(2, probe.rows());
            assertEquals(3, probe.columns());
            assertEquals(2, probe.sampledRows());
            assertEquals(-8.411765f, probe.rowDotChecksum(), 0.00001f);
            assertTrue(probe.rowDotNanos() >= 0);
            assertEquals(2, probe.matVecRows());
            assertEquals(-8.411765f, probe.matVecChecksum(), 0.00001f);
            assertTrue(probe.matrixCacheNanos() >= 0);
            assertTrue(probe.matVecNanos() >= 0);
            assertTrue(probe.cachedMatVecNanos() >= 0);
            assertFalse(probe.preparedMatVecProbe());
            assertFalse(probe.preparedMatVecReady());
            assertTrue(probe.compactSummary().contains("tensor=blk.0.attn_q.weight"));
            assertTrue(probe.compactSummary().contains("preparedMatVecReady=false"));
        }
    }
}
