package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Shared GGUF runtime-profile fixtures and assertions for synthetic decoder layouts.
 */
final class GgufProfFx {
    private GgufProfFx() {
    }

    static Map<String, Object> llamaMetadata() {
        return Map.of(
                "general.architecture", "llama",
                "llama.block_count", 1);
    }

    static GGUFModel model(Map<String, Object> metadata, List<GGUFTensorInfo> tensors) {
        return new GGUFModel(3, metadata, tensors, 0, MemorySegment.NULL, null);
    }

    static GGUFTensorInfo tensor(String name, int typeId, long sizeInBytes, long... shape) {
        long[] tensorShape = shape.length > 0 ? shape : new long[]{1};
        return new GGUFTensorInfo(name, tensorShape, typeId, 0, sizeInBytes);
    }

    static List<GGUFTensorInfo> llamaDecoderTensors(int typeId, long sizeInBytes, long columns) {
        return List.of(
                tensor("token_embd.weight", typeId, sizeInBytes, columns, 256),
                tensor("output_norm.weight", 0, 16, columns),
                tensor("blk.0.attn_norm.weight", 0, 16, columns),
                tensor("blk.0.attn_q.weight", typeId, sizeInBytes, columns, 256),
                tensor("blk.0.attn_k.weight", typeId, sizeInBytes, columns, 256),
                tensor("blk.0.attn_v.weight", typeId, sizeInBytes, columns, 256),
                tensor("blk.0.attn_output.weight", typeId, sizeInBytes, columns, 256),
                tensor("blk.0.ffn_norm.weight", 0, 16, columns),
                tensor("blk.0.ffn_gate.weight", typeId, sizeInBytes, columns, 256),
                tensor("blk.0.ffn_up.weight", typeId, sizeInBytes, columns, 256),
                tensor("blk.0.ffn_down.weight", typeId, sizeInBytes, columns, 256)
        );
    }

    static void assertSingleTypeReady(int typeId, long sizeInBytes, long columns, String summaryType) {
        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(
                model(llamaMetadata(), llamaDecoderTensors(typeId, sizeInBytes, columns)),
                1024,
                3);

        assertReady(profile);
        assertTrue(profile.compactTypeSummary(3).contains(summaryType));
    }

    static void assertReady(GgufRuntimeProfile profile) {
        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertTrue(profile.unknownTensorTypeIds().isEmpty());
        assertTrue(profile.unsupportedRowDotTypeIds().isEmpty());
        assertTrue(profile.rowDotPrimitivesReady());
        assertEquals("loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled",
                profile.javaStatus());
    }
}
