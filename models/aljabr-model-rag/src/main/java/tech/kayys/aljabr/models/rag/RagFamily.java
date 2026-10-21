package tech.kayys.aljabr.models.rag;

import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;

import java.util.List;

/**
 * Retrieval-Augmented Generation (RAG) model architecture mapping.
 */
public class RagFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "rag";
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.GELU; // Generator uses gelu
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("RagModel", "RagSequenceForGeneration", "RagTokenForGeneration");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("rag");
    }

    @Override
    public String embedTokensWeight() {
        return "generator.model.shared.weight"; // Generator token embeddings
    }

    @Override
    public String finalNormWeight() {
        return "generator.model.encoder.layernorm_embedding.weight";
    }

    @Override
    public String lmHeadWeight() {
        return "generator.lm_head.weight";
    }

    @Override
    public String layerQueryWeight(int i) {
        return "generator.model.decoder.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    @Override
    public String layerKeyWeight(int i) {
        return "generator.model.decoder.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return "generator.model.decoder.layers.%d.self_attn.v_proj.weight".formatted(i);
    }

    @Override
    public String layerOutputWeight(int i) {
        return "generator.model.decoder.layers.%d.self_attn.out_proj.weight".formatted(i);
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "generator.model.decoder.layers.%d.self_attn_layer_norm.weight".formatted(i);
    }

    @Override
    public String layerFfnGateWeight(int i) {
        return null; // RAG generator (BART) uses simple FFN, not SwiGLU, so no separate gate
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "generator.model.decoder.layers.%d.fc1.weight".formatted(i);
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "generator.model.decoder.layers.%d.fc2.weight".formatted(i);
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "generator.model.decoder.layers.%d.encoder_attn_layer_norm.weight".formatted(i); // Pre cross-attention norm
    }
    
    @Override
    public String layerPostFfnNormWeight(int i) {
        return "generator.model.decoder.layers.%d.final_layer_norm.weight".formatted(i);
    }

    @Override
    public boolean usesRmsNorm() {
        return false; // Uses LayerNorm
    }

    @Override
    public boolean hasSeparateGateProjection() {
        return false;
    }
}
