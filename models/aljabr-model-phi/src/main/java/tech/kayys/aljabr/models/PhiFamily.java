package tech.kayys.tafkir.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;
import java.util.List;

/**
 * Phi family (Microsoft).
 * Uses fused QKV and fused gate/up projection layouts.
 */
@ApplicationScoped
public class PhiFamily implements ModelArchitecture {
    @Override
    public String id() {
        return "phi";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("PhiForCausalLM", "Phi3ForCausalLM", "Phi3SmallForCausalLM", "Phi4ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("phi", "phi3", "phi4");
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
    public List<String> finalNormWeightCandidates() {
        return List.of("model.norm.weight", "model.final_layernorm.weight");
    }

    @Override
    public String finalNormBias() {
        return null;
    }

    @Override
    public boolean hasFusedQKV() {
        return true;
    }

    @Override
    public String layerFusedQKVWeight(int i) {
        return "model.layers.%d.self_attn.qkv_proj.weight".formatted(i);
    }

    @Override
    public String layerQueryWeight(int i) {
        return layerFusedQKVWeight(i);
    } // not used separately

    @Override
    public String layerKeyWeight(int i) {
        return layerFusedQKVWeight(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return layerFusedQKVWeight(i);
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
        return "model.layers.%d.mlp.gate_up_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers.%d.mlp.gate_up_proj.weight".formatted(i);
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
    public boolean usesRmsNorm() {
        return true;
    }

    @Override
    public ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return PhiRuntimeProfile.text(config);
    }
}
