package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.*;

class GgufPrepSmallTest {
    @Test
    void directSmallAndQ8PreparationCachesExactPreparedBytes() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(224);
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ8Block(segment.asSlice(18, 34), (short) 0x3c00, (byte) 1);
            writeNVFP4Block(segment.asSlice(52, 36), (byte) 0x40, (byte) 0xA5);
            writeIQ4XSBlock(segment.asSlice(88, 136), (short) 0x3c00, (byte) 0x88);
            GGUFTensorInfo q32 = new GGUFTensorInfo("q32.direct", new long[]{32, 1}, 2, 0, 18);
            GGUFTensorInfo q8 = new GGUFTensorInfo("q8.direct", new long[]{32, 1}, 8, 18, 34);
            GGUFTensorInfo nvfp4 = new GGUFTensorInfo("nvfp4.direct", new long[]{64, 1}, 40, 52, 36);
            GGUFTensorInfo iq4xs = new GGUFTensorInfo("iq4xs.direct", new long[]{256, 1}, 23, 88, 136);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(q32, q8, nvfp4, iq4xs), 0, segment, null);

            assertEquals(0, GgufTensorOps.preparedMatrixEstimateCacheSize(model));

            assertEquals(36L, GgufTensorOps.q32Matrix(model, q32).estimatedBytes());
            assertEquals(36L, GgufTensorOps.q8Matrix(model, q8).estimatedBytes());
            assertEquals(80L, GgufTensorOps.q8Matrix(model, nvfp4).estimatedBytes());
            assertEquals(288L, GgufTensorOps.q8Matrix(model, iq4xs).estimatedBytes());

            assertEquals(4, GgufTensorOps.preparedMatrixEstimateCacheSize(model));
            assertEquals(36L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q32));
            assertEquals(36L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, q8));
            assertEquals(80L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, nvfp4));
            assertEquals(288L, GgufTensorOps.estimatePreparedMatrixCacheBytes(model, iq4xs));
        }
    }
}
