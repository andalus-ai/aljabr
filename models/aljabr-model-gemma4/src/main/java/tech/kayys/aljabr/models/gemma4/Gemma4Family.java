package tech.kayys.aljabr.models.gemma4;

import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.util.List;

import jakarta.enterprise.context.ApplicationScoped;

/**
 * Google Gemma 4 text adapter. Multimodal wrappers stay guarded by runtime checks.
 */
@ApplicationScoped
public class Gemma4Family implements ModelArchitecture {
    private static final String TEXT_MODEL_PREFIX = "model.";
    private static final String UNIFIED_TEXT_MODEL_PREFIX = "model.language_model.";

    @Override
    public String id() {
        return "gemma4";
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.GELU;
    }

    @Override
    public boolean usesNeoxRope() {
        return true;
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
                "Gemma4ForImageTextToText", "Gemma4ForMultimodalLM",
                "Gemma4UnifiedForConditionalGeneration");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public List<String> embedTokensWeightCandidates() {
        return textDecoderCandidates(embedTokensWeight());
    }

    public String embedTokensPerLayerWeight() {
        return "model.embed_tokens_per_layer.weight";
    }

    @Override
    public List<String> embedTokensPerLayerWeightCandidates() {
        return textDecoderCandidates(embedTokensPerLayerWeight());
    }

    @Override
    public String perLayerModelProjectionWeight() {
        return "model.per_layer_model_projection.weight";
    }

    @Override
    public List<String> perLayerModelProjectionWeightCandidates() {
        return textDecoderCandidates(perLayerModelProjectionWeight());
    }

    @Override
    public String perLayerProjectionNormWeight() {
        return "model.per_layer_projection_norm.weight";
    }

    @Override
    public List<String> perLayerProjectionNormWeightCandidates() {
        return textDecoderCandidates(perLayerProjectionNormWeight());
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    @Override
    public List<String> finalNormWeightCandidates() {
        return textDecoderCandidates(finalNormWeight());
    }

    @Override
    public String lmHeadWeight() {
        return "lm_head.weight";
    }

    @Override
    public List<String> lmHeadWeightCandidates() {
        return List.of(lmHeadWeight(), "model.language_model.lm_head.weight", "language_model.lm_head.weight");
    }

    @Override
    public String layerQueryWeight(int i) {
        return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerQueryWeightCandidates(int i) {
        return textDecoderCandidates(layerQueryWeight(i));
    }

    @Override
    public String layerKeyWeight(int i) {
        return "model.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerKeyWeightCandidates(int i) {
        return textDecoderCandidates(layerKeyWeight(i));
    }

    @Override
    public String layerValueWeight(int i) {
        return "model.layers.%d.self_attn.v_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerValueWeightCandidates(int i) {
        return textDecoderCandidates(layerValueWeight(i));
    }

    @Override
    public String layerOutputWeight(int i) {
        return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerOutputWeightCandidates(int i) {
        return textDecoderCandidates(layerOutputWeight(i));
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "model.layers.%d.input_layernorm.weight".formatted(i);
    }

    @Override
    public List<String> layerAttentionNormWeightCandidates(int i) {
        return textDecoderCandidates(layerAttentionNormWeight(i));
    }

    @Override
    public String layerQueryNormWeight(int i) {
        return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
    }

    @Override
    public List<String> layerQueryNormWeightCandidates(int i) {
        return textDecoderCandidates(layerQueryNormWeight(i));
    }

    @Override
    public String layerKeyNormWeight(int i) {
        return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
    }

    @Override
    public List<String> layerKeyNormWeightCandidates(int i) {
        return textDecoderCandidates(layerKeyNormWeight(i));
    }

    @Override
    public String layerPostAttnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    @Override
    public List<String> layerPostAttnNormWeightCandidates(int i) {
        return textDecoderCandidates(layerPostAttnNormWeight(i));
    }

    @Override
    public String layerPreFfnNormWeight(int i) {
        return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public List<String> layerPreFfnNormWeightCandidates(int i) {
        return textDecoderCandidates(layerPreFfnNormWeight(i));
    }

    @Override
    public String layerPerLayerInputGateWeight(int i) {
        return "model.layers.%d.per_layer_input_gate.weight".formatted(i);
    }

    @Override
    public List<String> layerPerLayerInputGateWeightCandidates(int i) {
        return textDecoderCandidates(layerPerLayerInputGateWeight(i));
    }

    @Override
    public String layerPerLayerProjectionWeight(int i) {
        return "model.layers.%d.per_layer_projection.weight".formatted(i);
    }

    @Override
    public List<String> layerPerLayerProjectionWeightCandidates(int i) {
        return textDecoderCandidates(layerPerLayerProjectionWeight(i));
    }

    @Override
    public String layerPostPerLayerInputNormWeight(int i) {
        return "model.layers.%d.post_per_layer_input_norm.weight".formatted(i);
    }

    @Override
    public List<String> layerPostPerLayerInputNormWeightCandidates(int i) {
        return textDecoderCandidates(layerPostPerLayerInputNormWeight(i));
    }

    @Override
    public String layerScalarWeight(int i) {
        return "model.layers.%d.layer_scalar".formatted(i);
    }

    @Override
    public List<String> layerScalarWeightCandidates(int i) {
        return textDecoderCandidates(layerScalarWeight(i));
    }

    @Override
    public String layerFfnGateWeight(int i) {
        return "model.layers.%d.mlp.gate_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerFfnGateWeightCandidates(int i) {
        return textDecoderCandidates(layerFfnGateWeight(i));
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers.%d.mlp.up_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerFfnUpWeightCandidates(int i) {
        return textDecoderCandidates(layerFfnUpWeight(i));
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "model.layers.%d.mlp.down_proj.weight".formatted(i);
    }

    @Override
    public List<String> layerFfnDownWeightCandidates(int i) {
        return textDecoderCandidates(layerFfnDownWeight(i));
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public List<String> layerFfnNormWeightCandidates(int i) {
        return textDecoderCandidates(layerFfnNormWeight(i));
    }

    @Override
    public String layerPostFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public List<String> layerPostFfnNormWeightCandidates(int i) {
        return textDecoderCandidates(layerPostFfnNormWeight(i));
    }

    @Override
    public String layerMoeGateWeight(int i) {
        return "model.layers.%d.mlp.router.weight".formatted(i);
    }

    @Override
    public List<String> layerMoeGateWeightCandidates(int i) {
        return textDecoderCandidates(layerMoeGateWeight(i));
    }

    @Override
    public String expertGateWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.mlp.experts.%d.gate_proj.weight".formatted(layerIdx, expertIdx);
    }

    @Override
    public List<String> expertGateWeightCandidates(int layerIdx, int expertIdx) {
        return textDecoderCandidates(expertGateWeight(layerIdx, expertIdx));
    }

    @Override
    public String expertUpWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.mlp.experts.%d.up_proj.weight".formatted(layerIdx, expertIdx);
    }

    @Override
    public List<String> expertUpWeightCandidates(int layerIdx, int expertIdx) {
        return textDecoderCandidates(expertUpWeight(layerIdx, expertIdx));
    }

    @Override
    public String expertDownWeight(int layerIdx, int expertIdx) {
        return "model.layers.%d.mlp.experts.%d.down_proj.weight".formatted(layerIdx, expertIdx);
    }

    @Override
    public List<String> expertDownWeightCandidates(int layerIdx, int expertIdx) {
        return textDecoderCandidates(expertDownWeight(layerIdx, expertIdx));
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
        return Gemma4RuntimeProfile.text(config);
    }

    @Override
    public float defaultAttnSoftCap() {
        return 0.0f;
    }

    @Override
    public float defaultFinalSoftCap() {
        return 0.0f;
    }

    public String visionPatchEmbeddingWeight() {
        return "model.vision_tower.patch_embedder.input_proj.weight";
    }

    public String visionPatchEmbeddingBias() {
        return "model.vision_tower.patch_embedder.input_proj.bias";
    }

    public String visionPositionEmbeddingWeight() {
        return "model.vision_tower.patch_embedder.position_embedding_table";
    }

    public String visionMultiModalProjectorWeight() {
        return "model.multi_modal_projector.linear.weight";
    }

    public String visionMultiModalProjectorBias() {
        return "model.multi_modal_projector.linear.bias";
    }

    public String visionLayerQueryWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    public String visionLayerKeyWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    public String visionLayerValueWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.self_attn.v_proj.weight".formatted(i);
    }

    public String visionLayerOutputWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.self_attn.o_proj.weight".formatted(i);
    }

    public String visionLayerAttentionNormWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.input_layernorm.weight".formatted(i);
    }

    public String visionLayerFfnGateWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.mlp.gate_proj.weight".formatted(i);
    }

    public String visionLayerFfnUpWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.mlp.up_proj.weight".formatted(i);
    }

    public String visionLayerFfnDownWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.mlp.down_proj.weight".formatted(i);
    }

    public String visionLayerPostAttentionNormWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    public String visionLayerPreFeedforwardNormWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
    }

    public String visionLayerPostFeedforwardNormWeight(int i) {
        return "model.vision_tower.encoder.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    private static List<String> textDecoderCandidates(String canonicalName) {
        if (canonicalName == null || canonicalName.isBlank()) {
            return List.of();
        }
        String unifiedName = unifiedTextName(canonicalName);
        if (canonicalName.equals(unifiedName)) {
            return List.of(canonicalName);
        }
        return List.of(canonicalName, unifiedName);
    }

    private static String unifiedTextName(String canonicalName) {
        if (canonicalName.startsWith(TEXT_MODEL_PREFIX)) {
            return UNIFIED_TEXT_MODEL_PREFIX + canonicalName.substring(TEXT_MODEL_PREFIX.length());
        }
        return canonicalName;
    }
}
