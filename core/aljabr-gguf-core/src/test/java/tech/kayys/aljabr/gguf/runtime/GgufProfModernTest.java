package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.assertReady;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.llamaMetadata;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.model;
import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.tensor;

class GgufProfModernTest {
    @Test
    void reportsModernKnownQuantLayoutsAsJavaRowDotReady() {
        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 34, 54, 256, 256),
                tensor("output_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_q.weight", 34, 54, 256, 256),
                tensor("blk.0.attn_k.weight", 35, 66, 256, 256),
                tensor("blk.0.attn_v.weight", 39, 17, 256, 256),
                tensor("blk.0.attn_output.weight", 40, 36, 256, 256),
                tensor("blk.0.ffn_norm.weight", 0, 16, 256),
                tensor("blk.0.ffn_gate.weight", 41, 18, 256, 256),
                tensor("blk.0.ffn_up.weight", 34, 54, 256, 256),
                tensor("blk.0.ffn_down.weight", 35, 66, 256, 256)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(llamaMetadata(), tensors), 1024, 3);

        assertReady(profile);
        assertEquals(1.0d, profile.knownTensorTypeRatio(), 0.0001d);
    }
}
