package tech.kayys.tafkir.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import java.util.List;

/**
 * LLaMA-3 / 3.1 / 3.2 / 3.3 architecture (1B, 3B, 8B, 11B, 70B, 90B, 405B).
 *
 * Key differences from LLaMA-2:
 * - 128K vocabulary (tiktoken-based, vs 32K for LLaMA-2)
 * - GQA on all sizes (LLaMA-2 70B only)
 * - RoPE theta: 500000 (vs 10000 for LLaMA-2)
 * - LLaMA-3.1+: YaRN context extension to 128K
 * - LLaMA-3.2: 1B and 3B are smaller distillations; 11B/90B add vision
 * - LLaMA-3.3: 70B with improved instruction following
 *
 * Weight names: identical to LLaMA-1/2. Detected via arch class
 * LlamaForCausalLM
 * and vocab_size > 32000 or rope_theta > 100000 in config.json.
 *
 * HuggingFace models:
 * meta-llama/Llama-3.1-8B-Instruct
 * meta-llama/Llama-3.1-70B-Instruct
 * meta-llama/Llama-3.1-405B-Instruct-FP8
 * meta-llama/Llama-3.2-1B-Instruct
 * meta-llama/Llama-3.3-70B-Instruct
 */
@ApplicationScoped
public class LLaMA3Family implements ModelArchitecture {

    @Override
    public String id() {
        return "llama3";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("LlamaForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("llama", "llama3");
    }

    // Same weight names as LLaMA-1/2 — just different config values
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
        return "lm_head.weight";
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
}