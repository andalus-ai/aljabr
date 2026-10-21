package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Shared runtime-probe fixtures for compact one-tensor GGUF models.
 */
final class GgufProbeFx {
    private static final String DEFAULT_TENSOR = "blk.0.attn_q.weight";

    private GgufProbeFx() {
    }

    static GGUFModel oneTensorModel(MemorySegment segment, long[] shape, int type, long nbytes) {
        GGUFTensorInfo tensor = new GGUFTensorInfo(DEFAULT_TENSOR, shape, type, 0, nbytes);
        return new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
    }

    static GGUFModel decoderCacheModel(Arena arena) {
        MemorySegment segment = arena.allocate(52);
        GgufQ32Fx.writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
        GgufQFx.writeQ8Block(segment.asSlice(18, 34), (short) 0x3c00, (byte) 1);
        GGUFTensorInfo q4 = new GGUFTensorInfo(
                "blk.0.attn_q.weight",
                new long[]{32, 1},
                2,
                0,
                18);
        GGUFTensorInfo q8 = new GGUFTensorInfo(
                "blk.0.attn_v.weight",
                new long[]{32, 1},
                8,
                18,
                34);
        GGUFTensorInfo ignored = new GGUFTensorInfo(
                "token_embd.weight",
                new long[]{32, 1},
                8,
                18,
                34);
        return new GGUFModel(3, Map.of(), List.of(q4, q8, ignored), 0, segment, null);
    }

    static void assertPreparedMatrixProbe(
            GgufRuntimeProbe probe,
            String tensorType,
            int rows,
            int columns,
            float checksum) {
        assertTrue(probe.hasTensorProbe());
        assertEquals(tensorType, probe.tensorType());
        assertEquals(rows, probe.rows());
        assertEquals(columns, probe.columns());
        assertEquals(checksum, probe.rowDotChecksum(), 0.00001f);
        assertEquals(checksum, probe.matVecChecksum(), 0.00001f);
        assertEquals(probe.matVecChecksum(), probe.cachedMatVecChecksum(), 0.00001f);
        assertTrue(probe.preparedMatVecProbe());
        assertTrue(probe.preparedMatVecReady());
        assertTrue(probe.matVecChecksumsAgree());
        assertTrue(probe.matrixCacheNanos() >= 0);
        assertTrue(probe.cachedMatVecNanos() >= 0);
        assertTrue(probe.compactSummary().contains("type=" + tensorType));
        assertTrue(probe.compactSummary().contains("preparedMatVecReady=true"));
    }
}
