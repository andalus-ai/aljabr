package tech.kayys.aljabr.models.afmoe;

import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;

import java.util.List;

/**
 * AFMOE model architecture mapping.
 */
public class AfmoeFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "afmoe";
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.SILU;
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("AfmoeForCausalLM", "AfmoeModel");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("afmoe");
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

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    @Override
    public String layerMoeGateWeight(int i) {
        return "model.layers.%d.mlp.router.weight".formatted(i);
    }

    @Override
    public String expertGateWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.mlp.experts.%d.gate_proj.weight".formatted(layerIdx, expertIdx);
    }

    @Override
    public String expertUpWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.mlp.experts.%d.up_proj.weight".formatted(layerIdx, expertIdx);
    }

    @Override
    public String expertDownWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.mlp.experts.%d.down_proj.weight".formatted(layerIdx, expertIdx);
    }

    @Override
    public boolean usesRmsNorm() {
        return true;
    }

    @Override
    public boolean hasSeparateGateProjection() {
        return true;
    }
}
