package tech.kayys.aljabr.models.gemma3n;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Gemma3nModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gemma3n",
                "Google Gemma 3n",
                List.of("gemma3n", "gemma3n_text", "gemma3n_audio", "gemma3n_vision"),
                List.of("Gemma3nForConditionalGeneration", "Gemma3nForCausalLM", "Gemma3nModel",
                        "Gemma3nTextModel", "Gemma3nAudioEncoder"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "gemma_sentencepiece_with_audio_vision_processor",
                        "direct_safetensor", "pending_gemma3n_audio_vision_projector_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("gemma3n-spm-bpe"));
    }
}
