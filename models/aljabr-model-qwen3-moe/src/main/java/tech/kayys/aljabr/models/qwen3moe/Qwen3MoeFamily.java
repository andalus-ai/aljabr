package tech.kayys.aljabr.models.qwen3moe;

import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;

import java.util.List;
import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class Qwen3MoeFamily implements ModelArchitecture {
    
    @Override
    public String id() { return "qwen3_moe"; }
    
    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Qwen3MoeForCausalLM", "Qwen3MoeModel", "Qwen3MoeForSequenceClassification");
    }
    
    @Override
    public List<String> supportedModelTypes() {
        return List.of("qwen3_moe");
    }
    
    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public List<String> embedTokensWeightCandidates() {
        return List.of("model.embed_tokens.weight", "model.language_model.embed_tokens.weight");
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    @Override
    public List<String> finalNormWeightCandidates() {
        return List.of("model.norm.weight", "model.language_model.norm.weight");
    }

    @Override
    public String lmHeadWeight() {
        return "lm_head.weight";
    }

    @Override
    public String layerQueryWeight(int i) {
        return "model.layers." + i + ".self_attn.q_proj.weight";
    }

    @Override
    public List<String> layerQueryWeightCandidates(int i) {
        return List.of(layerQueryWeight(i), "model.language_model.layers." + i + ".self_attn.q_proj.weight");
    }

    @Override
    public String layerKeyWeight(int i) {
        return "model.layers." + i + ".self_attn.k_proj.weight";
    }

    @Override
    public List<String> layerKeyWeightCandidates(int i) {
        return List.of(layerKeyWeight(i), "model.language_model.layers." + i + ".self_attn.k_proj.weight");
    }

    @Override
    public String layerValueWeight(int i) {
        return "model.layers." + i + ".self_attn.v_proj.weight";
    }

    @Override
    public List<String> layerValueWeightCandidates(int i) {
        return List.of(layerValueWeight(i), "model.language_model.layers." + i + ".self_attn.v_proj.weight");
    }

    @Override
    public String layerOutputWeight(int i) {
        return "model.layers." + i + ".self_attn.o_proj.weight";
    }

    @Override
    public List<String> layerOutputWeightCandidates(int i) {
        return List.of(layerOutputWeight(i), "model.language_model.layers." + i + ".self_attn.o_proj.weight");
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "model.layers." + i + ".input_layernorm.weight";
    }

    @Override
    public List<String> layerAttentionNormWeightCandidates(int i) {
        return List.of(layerAttentionNormWeight(i), "model.language_model.layers." + i + ".input_layernorm.weight");
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers." + i + ".post_attention_layernorm.weight";
    }

    @Override
    public List<String> layerFfnNormWeightCandidates(int i) {
        return List.of(layerFfnNormWeight(i), "model.language_model.layers." + i + ".post_attention_layernorm.weight");
    }

    @Override
    public String layerFfnGateWeight(int i) {
        return "model.layers." + i + ".mlp.gate_proj.weight";
    }

    @Override
    public List<String> layerFfnGateWeightCandidates(int i) {
        return List.of(layerFfnGateWeight(i), "model.language_model.layers." + i + ".mlp.gate_proj.weight");
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers." + i + ".mlp.up_proj.weight";
    }

    @Override
    public List<String> layerFfnUpWeightCandidates(int i) {
        return List.of(layerFfnUpWeight(i), "model.language_model.layers." + i + ".mlp.up_proj.weight");
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "model.layers." + i + ".mlp.down_proj.weight";
    }

    @Override
    public List<String> layerFfnDownWeightCandidates(int i) {
        return List.of(layerFfnDownWeight(i), "model.language_model.layers." + i + ".mlp.down_proj.weight");
    }

    @Override
    public String layerMoeGateWeight(int i) {
        return "model.layers." + i + ".mlp.gate.weight";
    }

    @Override
    public List<String> layerMoeGateWeightCandidates(int i) {
        return List.of(layerMoeGateWeight(i), "model.language_model.layers." + i + ".mlp.gate.weight");
    }

    @Override
    public String expertGateWeight(int layerIdx, int expertIdx) {
        return "model.layers." + layerIdx + ".mlp.experts.gate_up_proj";
    }

    @Override
    public List<String> expertGateWeightCandidates(int layerIdx, int expertIdx) {
        return List.of(expertGateWeight(layerIdx, expertIdx), "model.language_model.layers." + layerIdx + ".mlp.experts.gate_up_proj");
    }

    @Override
    public String expertUpWeight(int layerIdx, int expertIdx) {
        return expertGateWeight(layerIdx, expertIdx);
    }
    
    @Override
    public List<String> expertUpWeightCandidates(int layerIdx, int expertIdx) {
        return expertGateWeightCandidates(layerIdx, expertIdx);
    }

    @Override
    public String expertDownWeight(int layerIdx, int expertIdx) {
        return "model.layers." + layerIdx + ".mlp.experts.down_proj";
    }

    @Override
    public List<String> expertDownWeightCandidates(int layerIdx, int expertIdx) {
        return List.of(expertDownWeight(layerIdx, expertIdx), "model.language_model.layers." + layerIdx + ".mlp.experts.down_proj");
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
    public boolean usesNeoxRope() {
        return true;
    }
}
