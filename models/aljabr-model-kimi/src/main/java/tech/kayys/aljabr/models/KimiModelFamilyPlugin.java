package tech.kayys.tafkir.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class KimiModelFamilyPlugin implements ModelFamilyPlugin {

        @Override
        public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                                "kimi",
                                "Moonshot Kimi",
                                List.of("kimi", "kimi_k2"),
                                List.of("KimiForCausalLM", "KimiK2ForCausalLM"),
                                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                                                ModelFamilyCapability.CHAT_TEMPLATE),
                                Map.of(
                                                "bundle_profile", "experimental",
                                                "direct_safetensor", "pending_architecture_adapter",
                                                "tokenizer", "pending",
                                                "tokenizer_metadata_status", "pending",
                                                "tokenizer_metadata_pending_reason",
                                                "tokenizer adapter pending descriptor stabilization",
                                                "version", "0.1.0-SNAPSHOT"));
        }
}
