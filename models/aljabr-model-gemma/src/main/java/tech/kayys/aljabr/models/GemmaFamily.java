package tech.kayys.tafkir.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;
import java.util.List;

/**
 * Base Gemma text architecture adapter owned by the core Gemma model-family
 * module.
 *
 * <p>
 * Gemma 4 variants are intentionally owned by the dedicated Gemma 4 module so
 * direct SafeTensor resolution cannot depend on CDI or ServiceLoader ordering.
 * </p>
 */
@ApplicationScoped
public class GemmaFamily implements ModelArchitecture {
    @Override
    public String id() {
        return "gemma";
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.GELU;
    }

    @Override
    public boolean usesNeoxRope() {
        // Gemma text variants (including Gemma3) follow split-half rotate_half.
        return true;
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("GemmaForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    public String embedTokensPerLayerWeight() {
        return "model.embed_tokens_per_layer.weight";
    }

    @Override
    public String perLayerModelProjectionWeight() {
        return "model.per_layer_model_projection.weight";
    }

    @Override
    public String perLayerProjectionNormWeight() {
        return "model.per_layer_projection_norm.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
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

    // QK-norms (Gemma-3/4 apply RMSNorm to Q and K before attention)
    @Override
    public String layerQueryNormWeight(int i) {
        return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
    }

    @Override
    public String layerKeyNormWeight(int i) {
        return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
    }

    // Post-attention norm (Gemma-2/3/4)
    @Override
    public String layerPostAttnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    // Pre-FFN norm (Gemma-2/4)
    @Override
    public String layerPreFfnNormWeight(int i) {
        return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
    }

    // Per-layer input gating (Gemma-4)
    @Override
    public String layerPerLayerInputGateWeight(int i) {
        return "model.layers.%d.per_layer_input_gate.weight".formatted(i);
    }

    @Override
    public String layerPerLayerProjectionWeight(int i) {
        return "model.layers.%d.per_layer_projection.weight".formatted(i);
    }

    @Override
    public String layerPostPerLayerInputNormWeight(int i) {
        return "model.layers.%d.post_per_layer_input_norm.weight".formatted(i);
    }

    @Override
    public String layerScalarWeight(int i) {
        return "model.layers.%d.layer_scalar".formatted(i);
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
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public String layerPostFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    // ── Runtime inference behaviors ──────────────────────────────────────────

    @Override
    public float embeddingScaleFactor(int hiddenDim) {
        // Gemma scales embeddings by sqrt(hidden_dim)
        return (float) Math.sqrt(hiddenDim);
    }

    @Override
    public boolean addOneToRmsNormWeight() {
        return true;
    }

    @Override
    public ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return GemmaRuntimeProfile.text(config);
    }

    @Override
    public float defaultAttnSoftCap() {
        return 0.0f;
    }

    @Override
    public float defaultFinalSoftCap() {
        return 0.0f;
    }
}
