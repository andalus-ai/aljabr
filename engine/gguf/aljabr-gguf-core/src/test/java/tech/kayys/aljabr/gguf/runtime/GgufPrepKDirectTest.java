package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.*;

class GgufPrepKDirectTest {
    @Test
    void directNoMinKQuantPreparationCachesExactPreparedBytes() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(84 + 144 + 176);
            writeQ2KBlock(segment.asSlice(0, 84), (byte) 0x01, (byte) 0x55);
            writeSimpleQ4KBlock(segment.asSlice(84, 144));
            writeQ5KBlock(segment.asSlice(228, 176), (byte) 0xFF, (byte) 0);
            GGUFTensorInfo q2 = new GGUFTensorInfo("q2.direct.no_mins", new long[]{256, 1}, 10, 0, 84);
            GGUFTensorInfo q4 = new GGUFTensorInfo("q4.direct.no_mins", new long[]{256, 1}, 12, 84, 144);
            GGUFTensorInfo q5 = new GGUFTensorInfo("q5.direct.no_mins", new long[]{256, 1}, 13, 228, 176);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q2, q4, q5), 0, segment, null);

            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            assertEquals(320L, GgufTensorOps.q2KMatrix(model, q2).estimatedBytes());
            assertEquals(288L, GgufTensorOps.q4KMatrix(model, q4).estimatedBytes());
            assertEquals(288L, GgufTensorOps.q5KMatrix(model, q5).estimatedBytes());

            assertEquals(3, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(320L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q2));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q4));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q5));
        }
    }

    @Test
    void directPlainKQuantPreparationCachesExactPreparedBytes() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(110 + 210);
            writeQ3KBlock(segment.asSlice(0, 110), 1, (byte) 0x55, (byte) 0xFF);
            writeQ6KBlock(segment.asSlice(110, 210));
            GGUFTensorInfo q3 = new GGUFTensorInfo("q3.direct.plain", new long[]{256, 1}, 11, 0, 110);
            GGUFTensorInfo q6 = new GGUFTensorInfo("q6.direct.plain", new long[]{256, 1}, 14, 110, 210);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q3, q6), 0, segment, null);

            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            assertEquals(320L, GgufTensorOps.q3KMatrix(model, q3).estimatedBytes());
            assertEquals(320L, GgufTensorOps.q6KMatrix(model, q6).estimatedBytes());

            assertEquals(2, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(320L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q3));
            assertEquals(320L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q6));
        }
    }
}
