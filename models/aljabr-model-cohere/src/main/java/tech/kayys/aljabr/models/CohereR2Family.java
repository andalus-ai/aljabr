package tech.kayys.tafkir.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Cohere Command-R+ v2 / Cohere-2 architecture.
 *
 * New features vs original Command-R:
 * - Cohere-2: parallel SwiGLU FFN (added gate_proj)
 * - Larger context: 128K tokens
 * - Improved RAG grounding via tool use tokens
 *
 * HuggingFace models:
 * CohereForAI/c4ai-command-r-plus-08-2024
 * CohereForAI/c4ai-command-r7b-12-2024
 */
@ApplicationScoped
public class CohereR2Family implements ModelArchitecture {

    @Override
    public String id() {
        return "cohere2";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Cohere2ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("cohere2");
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
    } // tied

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
        return "model.layers.%d.input_layernorm.weight".formatted(i);
    }

    // Cohere-2 has gate_proj (unlike Cohere-1)
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
