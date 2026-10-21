package tech.kayys.aljabr.models.gemma3;

import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.util.List;

/**
 * Google Gemma 3 text architecture adapter.
 */
public class Gemma3TextFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "gemma3_text";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Gemma3ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma3", "gemma3_text");
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.GELU;
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
        return "model.embed_tokens.weight";
    }

    @Override
    public boolean hasTiedEmbeddings() {
        return true;
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
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public String layerPostFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public String layerPreFfnNormWeight(int i) {
        return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
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
    public String layerQueryNormWeight(int i) {
        return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
    }

    @Override
    public String layerKeyNormWeight(int i) {
        return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
    }

    @Override
    public String layerPostAttnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    public String layerQNorm(int i) {
        return layerQueryNormWeight(i);
    }

    public String layerKNorm(int i) {
        return layerKeyNormWeight(i);
    }

    public String layerPostAttnNorm(int i) {
        return layerPostAttnNormWeight(i);
    }

    @Override
    public float embeddingScaleFactor(int hiddenDim) {
        return (float) Math.sqrt(hiddenDim);
    }

    @Override
    public boolean addOneToRmsNormWeight() {
        return true;
    }

    @Override
    public ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return Gemma3RuntimeProfile.text(config);
    }

    @Override
    public boolean usesNeoxRope() {
        return true;
    }
}
