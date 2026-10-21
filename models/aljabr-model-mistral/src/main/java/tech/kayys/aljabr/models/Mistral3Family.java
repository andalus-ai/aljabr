package tech.kayys.tafkir.models;

import tech.kayys.aljabr.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Base Mistral architecture adapter retained in the core bundle.
 * Newer Mistral 3/4 and Ministral variants are detachable metadata plugins.
 */
public class Mistral3Family implements ModelArchitecture {

    @Override
    public String id() {
        return "mistral";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("MistralForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("mistral");
    }

    // Same weight names as Mistral-7B
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
