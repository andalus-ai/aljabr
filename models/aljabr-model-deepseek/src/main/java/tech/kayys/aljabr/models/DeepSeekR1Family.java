package tech.kayys.tafkir.models;

import tech.kayys.aljabr.spi.model.ModelArchitecture;
import java.util.List;

/**
 * DeepSeek-R1 compatibility architecture.
 *
 * R1 distillations reuse Qwen/Llama base model families; the flagship R1 line
 * reuses DeepSeek-V3-style weights, so this adapter only claims the R1 alias.
 */
public class DeepSeekR1Family implements ModelArchitecture {

    @Override
    public String id() {
        return "deepseek_r1";
    }

    @Override
    public List<String> supportedArchClassNames() {
        // R1 distillations reuse existing arch classes
        return List.of("DeepseekV3ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("deepseek_r1");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    @Override
    public String lmHeadWeight() {
        return "lm_head.weight";
    }

    @Override
    public String layerQueryWeight(int i) {
        return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    @Override
    public String layerKeyWeight(int i) {
        return "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight".formatted(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return "model.layers.%d.self_attn.kv_b_proj.weight".formatted(i);
    }

    @Override
    public String layerOutputWeight(int i) {
        return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "model.layers.%d.input_layernorm.weight".formatted(i);
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    @Override
    public String layerFfnGateWeight(int i) {
        return "model.layers.%d.mlp.gate_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers.%d.mlp.up_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "model.layers.%d.mlp.down_proj.weight".formatted(i);
    }
}
