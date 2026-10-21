package tech.kayys.aljabr.models.t5gemma2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class T5Gemma2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "t5gemma2",
                "Google T5Gemma 2",
                List.of("t5gemma2", "t5gemma2_text", "t5gemma2_encoder", "t5gemma2_decoder"),
                List.of("T5Gemma2ForConditionalGeneration", "T5Gemma2Model", "T5Gemma2TextEncoder",
                        "T5Gemma2Encoder", "T5Gemma2Decoder", "T5Gemma2ForSequenceClassification",
                        "T5Gemma2ForTokenClassification"),
                List.of(
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "gemma_sentencepiece_seq2seq_with_optional_vision_projector",
                        "direct_safetensor", "pending_t5gemma2_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("t5gemma2-spm-bpe"));
    }
}
