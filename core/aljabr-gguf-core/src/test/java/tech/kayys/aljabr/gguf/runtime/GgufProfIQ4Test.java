package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.assertReady;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.llamaMetadata;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.model;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.tensor;

class GgufProfIQ4Test {
    @Test
    void treatsIQ4DecoderTensorsAsJavaRowDotReady() {
        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 20, 18, 32, 256),
                tensor("output_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_q.weight", 23, 136, 256, 256),
                tensor("blk.0.attn_k.weight", 20, 18, 32, 256),
                tensor("blk.0.attn_v.weight", 23, 136, 256, 256),
                tensor("blk.0.attn_output.weight", 23, 136, 256, 256),
                tensor("blk.0.ffn_norm.weight", 0, 16, 256),
                tensor("blk.0.ffn_gate.weight", 23, 136, 256, 256),
                tensor("blk.0.ffn_up.weight", 23, 136, 256, 256),
                tensor("blk.0.ffn_down.weight", 23, 136, 256, 256)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(llamaMetadata(), tensors), 1024, 3);

        assertReady(profile);
        assertTrue(profile.compactTypeSummary(3).contains("IQ4_XS"));
    }
}
