package tech.kayys.tafkir.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Yi / Yi-1.5 architecture (6B, 9B, 34B).
 *
 * LLaMA-2-style architecture with different tokenizer and config.
 * Uses 4096 sequence length by default, extended to 200K via RoPE scaling.
 *
 * HuggingFace models:
 * 01-ai/Yi-1.5-6B-Chat, 01-ai/Yi-1.5-9B-Chat
 * 01-ai/Yi-1.5-34B-Chat
 */
@ApplicationScoped
public class YiFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "yi";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("YiForCausalLM", "LlamaForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("yi", "yi_1_5");
    }

    // Yi uses LlamaForCausalLM weights — identical naming
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
        return "model.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return "model.layers.%d.self_attn.v_proj.weight".formatted(i);
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
