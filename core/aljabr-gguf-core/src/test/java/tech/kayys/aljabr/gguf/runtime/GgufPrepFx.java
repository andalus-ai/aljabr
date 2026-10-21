package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_0Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeQ8Block;

/**
 * Shared prepared-matrix cache model fixtures for GGUF runtime tests.
 */
final class GgufPrepFx {
    private GgufPrepFx() {
    }

    static GGUFModel mixedPreparedPlanModel(MemorySegment segment) {
        return mixedPreparedModel(segment, false);
    }

    static GGUFModel mixedPreparedModel(MemorySegment segment) {
        return mixedPreparedModel(segment, true);
    }

    private static GGUFModel mixedPreparedModel(MemorySegment segment, boolean writeData) {
        if (writeData) {
            writeQ4_0Block(segment.asSlice(0, 18), (short) 0x3c00, (byte) 0x98);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(18, 144));
            writeQ8Block(segment.asSlice(162, 34), (short) 0x3c00, (byte) 1);
        }

        GGUFTensorInfo q4_0 = new GGUFTensorInfo("blk.0.attn_q.weight", new long[]{32, 1}, 2, 0, 18);
        GGUFTensorInfo q4K = new GGUFTensorInfo("blk.0.attn_k.weight", new long[]{256, 1}, 12, 18, 144);
        GGUFTensorInfo q8 = new GGUFTensorInfo("blk.0.attn_v.weight", new long[]{32, 1}, 8, 162, 34);
        GGUFTensorInfo f32 = new GGUFTensorInfo("blk.0.attn_output.weight", new long[]{2, 1}, 0, 0, 8);
        GGUFTensorInfo norm = new GGUFTensorInfo("blk.0.attn_norm.weight", new long[]{32}, 0, 0, 128);
        return new GGUFModel(3, Map.of(), List.of(q4_0, q4K, q8, f32, norm), 0, segment, null);
    }
}
