package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.model;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.tensor;

class GgufProfDiagTest {
    @Test
    void requiresGemma4AttentionNormTensorsBeforeDeclaringDecoderReady() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.block_count", 1);

        List<GGUFTensorInfo> tensors = new ArrayList<>();
        tensors.add(tensor("token_embd.weight", 0, 16));
        tensors.add(tensor("output_norm.weight", 0, 16));
        tensors.add(tensor("blk.0.attn_norm.weight", 0, 16));
        tensors.add(tensor("blk.0.attn_q.weight", 12, 144));
        tensors.add(tensor("blk.0.attn_k.weight", 12, 144));
        tensors.add(tensor("blk.0.attn_v.weight", 12, 144));
        tensors.add(tensor("blk.0.attn_output.weight", 12, 144));
        tensors.add(tensor("blk.0.ffn_norm.weight", 0, 16));
        tensors.add(tensor("blk.0.ffn_gate.weight", 12, 144));
        tensors.add(tensor("blk.0.ffn_up.weight", 12, 144));
        tensors.add(tensor("blk.0.ffn_down.weight", 12, 144));

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1280, 7);

        assertEquals(13, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertEquals(2, profile.missingDecoderTensorCount());
        assertEquals(0, profile.malformedDecoderTensorCount());
        assertEquals("loader-ready; decoder-tensors-missing=2; generation-disabled", profile.javaStatus());
        assertTrue(profile.missingDecoderTensorExamples().contains("blk.0.attn_q_norm.weight"));
        assertTrue(profile.missingDecoderTensorExamples().contains("blk.0.attn_k_norm.weight"));
    }

    @Test
    void reportsMissingDecoderTensorsAndUnknownTypes() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.block_count", 1);

        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 0, 16),
                tensor("output_norm.weight", 0, 16),
                tensor("blk.0.attn_norm.weight", 99, 16)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 64, 1);

        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(3, profile.presentDecoderTensorCount());
        assertTrue(profile.knownTensorTypeRatio() < 1.0d);
        assertEquals(List.of("99"), profile.unknownTensorTypeIds());
        assertEquals(List.of("99"), profile.unsupportedRowDotTypeIds());
        assertEquals("loader-ready; unknown-tensor-types=99; generation-disabled", profile.javaStatus());
        assertTrue(profile.missingDecoderTensorExamples().contains("blk.0.attn_q.weight"));
    }
}
