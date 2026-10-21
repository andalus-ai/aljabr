package tech.kayys.aljabr.models.gemma2;

import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;

import java.util.List;

/**
 * Google Gemma 2 architecture adapter.
 */
public class Gemma2Family implements ModelArchitecture {

    @Override
    public String id() {
        return "gemma2";
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.GELU;
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Gemma2ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma2");
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
    public String layerPreFfnNormWeight(int i) {
        return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
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
    public String layerPostAttnNormWeight(int i) {
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

    @Override
    public float embeddingScaleFactor(int hiddenDim) {
        return (float) Math.sqrt(hiddenDim);
    }

    @Override
    public boolean usesNeoxRope() {
        return false;
    }

    @Override
    public float defaultAttnSoftCap() {
        return 50.0f;
    }

    @Override
    public float defaultFinalSoftCap() {
        return 30.0f;
    }
}
