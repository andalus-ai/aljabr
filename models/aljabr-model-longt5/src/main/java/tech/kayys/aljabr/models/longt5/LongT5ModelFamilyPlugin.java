package tech.kayys.aljabr.models.longt5;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LongT5ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "longt5",
                "LongT5",
                List.of("longt5"),
                List.of("LongT5ForConditionalGeneration", "LongT5Model", "LongT5EncoderModel"),
                List.of(
                        ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_longt5_global_local_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(sentencePieceUnigram("longt5-sentencepiece-unigram"));
    }

    private static ModelTokenizerDescriptor sentencePieceUnigram(String id) {
        return new ModelTokenizerDescriptor(
                id,
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("spiece.model"),
                        List.of("sentencepiece.model"), List.of("tokenizer.model"),
                        List.of("tokenizer/tokenizer.json"), List.of("tokenizer/spiece.model"),
                        List.of("tokenizer/sentencepiece.model"), List.of("tokenizer/tokenizer.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_unigram",
                        "status", "metadata_only_until_unigram_tokenizer_runtime"));
    }
}
