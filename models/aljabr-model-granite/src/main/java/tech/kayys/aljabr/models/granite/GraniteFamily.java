package tech.kayys.aljabr.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.util.List;

/**
 * IBM Granite (dense decoder) family — {@code GraniteForCausalLM}.
 *
 * <p>Granite 3.x / 4.x uses the same weight naming layout as LLaMA but with
 * several architecture-level knobs that differ from vanilla LLaMA:
 *
 * <ul>
 *   <li>{@code embedding_multiplier} — scales the token embedding lookup before
 *       the first transformer block (analogous to Gemma's sqrt(hidden_dim) scale
 *       but it is a fixed constant stored in config.json).</li>
 *   <li>{@code attention_multiplier} — scales the attention logits instead of the
 *       standard {@code 1/sqrt(head_dim)} softmax temperature.</li>
 *   <li>{@code residual_multiplier} — scales every residual add.</li>
 *   <li>{@code logits_scaling} — divides the final pre-softmax logits.</li>
 *   <li>{@code tie_word_embeddings: true} — no separate {@code lm_head.weight}.</li>
 *   <li>{@code rope_theta: 10_000_000} — much larger than LLaMA's 10 000.</li>
 * </ul>
 *
 * <p>The weight naming follows the LLaMA HuggingFace convention exactly, so
 * all {@code model.layers.*} paths are identical to {@link LlamaFamily}.
 *
 * <p>Supports dense Granite variants only. GraniteMoe / GraniteMoeHybrid require
 * additional expert-weight entries and are registered separately.
 */
@ApplicationScoped
public class GraniteFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "granite";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("GraniteForCausalLM", "GraniteModel");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("granite");
    }

    // ── Embedding / Head ──────────────────────────────────────────────────────

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    /**
     * Granite ties embeddings to the LM head — there is no separate
     * {@code lm_head.weight} tensor in the checkpoint.
     */
    @Override
    public String lmHeadWeight() {
        return null;
    }

    @Override
    public boolean hasTiedEmbeddings() {
        return true;
    }

    // ── Attention weights ─────────────────────────────────────────────────────

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

    // ── FFN weights ───────────────────────────────────────────────────────────

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

    // ── Architecture properties ───────────────────────────────────────────────

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.SILU;
    }

    @Override
    public boolean usesRmsNorm() {
        return true;
    }

    @Override
    public boolean hasSeparateGateProjection() {
        return true;
    }

    /**
     * Granite uses NeoX-style (split-half) RoPE, same as LLaMA 3 family.
     */
    @Override
    public boolean usesNeoxRope() {
        return true;
    }

    /**
     * Granite 4.1 sets {@code rope_theta: 10_000_000} in config.json.
     * The engine reads this from config when available; this default is the
     * fallback for older checkpoints that omit the field.
     */
    @Override
    public float defaultRopeFreqBase() {
        return 10_000_000.0f;
    }

    @Override
    public double rmsNormEps() {
        return 1e-5;
    }

    // ── Runtime traits ────────────────────────────────────────────────────────

    @Override
    public ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        return ModelRuntimeTraits.fallbackFromConfig(config);
    }

    // ── GGUF arch ─────────────────────────────────────────────────────────────

    @Override
    public boolean matchesGgufArch(String ggufArch) {
        return "granite".equals(ggufArch);
    }
}
