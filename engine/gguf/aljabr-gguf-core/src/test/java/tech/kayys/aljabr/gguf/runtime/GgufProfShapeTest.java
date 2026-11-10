package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.model;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.tensor;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufProfShapeTest {
    @Test
    void rejectsDecoderProjectionShapeDriftWhenMetadataProvidesDimensions() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.block_count", 1);
        metadata.put("gemma4.embedding_length", 2304);
        metadata.put("gemma4.attention.head_count", 8);
        metadata.put("gemma4.attention.head_count_kv", 4);

        List<GGUFTensorInfo> tensors = new ArrayList<>();
        tensors.add(tensor("token_embd.weight", 0, 16, 2304, 256));
        tensors.add(tensor("output_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.attn_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.attn_q.weight", 12, 144, 1024, 2304));
        tensors.add(tensor("blk.0.attn_k.weight", 12, 144, 2304, 1152));
        tensors.add(tensor("blk.0.attn_v.weight", 12, 144, 2304, 1152));
        tensors.add(tensor("blk.0.attn_q_norm.weight", 0, 16, 288));
        tensors.add(tensor("blk.0.attn_k_norm.weight", 0, 16, 288));
        tensors.add(tensor("blk.0.attn_output.weight", 12, 144, 2304, 2304));
        tensors.add(tensor("blk.0.ffn_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.ffn_gate.weight", 12, 144, 2304, 9216));
        tensors.add(tensor("blk.0.ffn_up.weight", 12, 144, 2304, 9216));
        tensors.add(tensor("blk.0.ffn_down.weight", 12, 144, 9216, 2304));

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1280, 7);

        assertEquals(13, profile.requiredDecoderTensorCount());
        assertEquals(12, profile.presentDecoderTensorCount());
        assertEquals(0, profile.missingDecoderTensorCount());
        assertEquals(1, profile.malformedDecoderTensorCount());
        assertEquals("loader-ready; decoder-tensor-shapes-invalid=1; generation-disabled", profile.javaStatus());
        assertTrue(profile.missingDecoderTensorExamples().isEmpty());
        assertTrue(profile.malformedDecoderTensorExamples().stream()
                .anyMatch(example -> example.startsWith("blk.0.attn_q.weight shape=")));
    }
}
