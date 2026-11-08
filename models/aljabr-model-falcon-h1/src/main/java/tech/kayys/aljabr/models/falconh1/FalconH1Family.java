package tech.kayys.aljabr.models.falconh1;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.util.List;

/**
 * TII Falcon-H1 architecture (Falcon-H1-Tiny-90M, Falcon-H1-1.5B, Falcon-H1-7B, etc.).
 *
 * Hybrid State Space Model (Mamba2) + Attention architecture.
 *
 * @author bhangun
 */
@ApplicationScoped
public class FalconH1Family implements ModelArchitecture {

    @Override
    public String id() {
        return "falcon_h1";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("FalconH1ForCausalLM", "FalconH1Model");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("falcon_h1", "falcon-h1");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.final_layernorm.weight";
    }

    @Override
    public String lmHeadWeight() {
        // Tied to embed_tokens
        return "model.embed_tokens.weight";
    }

    @Override
    public List<String> lmHeadWeightCandidates() {
        return List.of("lm_head.weight", "model.embed_tokens.weight");
    }

    @Override
    public boolean hasTiedEmbeddings() {
        return true;
    }

    // ── Attention weights ──

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

    // ── FFN weights ──

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.pre_ff_layernorm.weight".formatted(i);
    }

    @Override
    public String layerFfnGateWeight(int i) {
        return "model.layers.%d.feed_forward.gate_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers.%d.feed_forward.up_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "model.layers.%d.feed_forward.down_proj.weight".formatted(i);
    }

    // ── Mamba weights ──

    @Override
    public String layerMambaInProjWeight(int i) {
        return "model.layers.%d.mamba.in_proj.weight".formatted(i);
    }

    @Override
    public String layerMambaConv1dWeight(int i) {
        return "model.layers.%d.mamba.conv1d.weight".formatted(i);
    }

    @Override
    public String layerMambaConv1dBias(int i) {
        return "model.layers.%d.mamba.conv1d.bias".formatted(i);
    }

    @Override
    public String layerMambaDtBias(int i) {
        return "model.layers.%d.mamba.dt_bias".formatted(i);
    }

    @Override
    public String layerMambaALogWeight(int i) {
        return "model.layers.%d.mamba.A_log".formatted(i);
    }

    @Override
    public String layerMambaDWeight(int i) {
        return "model.layers.%d.mamba.D".formatted(i);
    }

    @Override
    public String layerMambaOutProjWeight(int i) {
        return "model.layers.%d.mamba.out_proj.weight".formatted(i);
    }

    @Override
    public String layerMambaNormWeight(int i) {
        return "model.layers.%d.mamba.norm.weight".formatted(i);
    }

    @Override
    public boolean isHybridAttentionMamba() {
        return true;
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.SILU;
    }

    @Override
    public boolean usesRmsNorm() {
        return true;
    }

    @Override
    public float defaultRopeFreqBase() {
        return 100000000000.0f;
    }

    @Override
    public float embeddingScaleFactor(int hiddenDim) {
        return 1.0f;
    }
}
