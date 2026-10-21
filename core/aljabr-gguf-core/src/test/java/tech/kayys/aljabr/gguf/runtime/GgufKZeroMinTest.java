package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.LE_SHORT;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeSimpleQ4KBlock;

class GgufKZeroMinTest {
    @Test
    void kQuantBuildersTreatZeroMinCodesAsNoMinsBeforeApplyingDMin() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeSimpleQ4KBlock(segment.asSlice(84, 144));
            writeQ5KBlock(segment.asSlice(228, 176), (byte) 0xFF, (byte) 0);
            segment.set(LE_SHORT, 82, (short) 0x7e00);
            segment.set(LE_SHORT, 86, (short) 0x7e00);
            segment.set(LE_SHORT, 230, (short) 0x7e00);
            GGUFTensorInfo q2 = new GGUFTensorInfo("q2.nan_dmin.no_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4 = new GGUFTensorInfo("q4.nan_dmin.no_mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5 = new GGUFTensorInfo("q5.nan_dmin.no_mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2, q4, q5), 0, segment, null);

            assertEquals(320L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q2));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q4));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q5));

            GgufTensorOps.Q2KMatrix q2Matrix = GgufTensorOps.q2KMatrix(model, q2);
            GgufTensorOps.Q4KMatrix q4Matrix = GgufTensorOps.q4KMatrix(model, q4);
            GgufTensorOps.Q5KMatrix q5Matrix = GgufTensorOps.q5KMatrix(model, q5);

            assertFalse(q2Matrix.hasGroupMins());
            assertFalse(q4Matrix.hasGroupMins());
            assertFalse(q5Matrix.hasGroupMins());
            assertEquals(0, q2Matrix.groupMins().length);
            assertEquals(0, q4Matrix.groupMins().length);
            assertEquals(0, q5Matrix.groupMins().length);
            assertEquals(320L, q2Matrix.estimatedBytes());
            assertEquals(288L, q4Matrix.estimatedBytes());
            assertEquals(288L, q5Matrix.estimatedBytes());
        }
    }
}
