package tech.kayys.aljabr.models.qwen2moe;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen2MoeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen2_moe",
                "Alibaba Qwen2-MoE",
                List.of("qwen2_moe"),
                List.of("Qwen2MoeForCausalLM", "Qwen2MoeModel",
                        "Qwen2MoeForSequenceClassification", "Qwen2MoeForTokenClassification",
                        "Qwen2MoeForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "qwen_huggingface_bpe",
                        "direct_safetensor", "pending_qwen2_moe_expert_routing_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("qwen2-moe-hf-bpe"));
    }
}
