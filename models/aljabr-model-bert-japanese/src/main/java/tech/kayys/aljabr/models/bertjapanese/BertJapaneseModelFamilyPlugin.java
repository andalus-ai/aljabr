package tech.kayys.aljabr.models.bertjapanese;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BertJapaneseModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bert_japanese",
                "BERT Japanese",
                List.of("bert_japanese", "bert-japanese"),
                List.of("BertModel", "BertForMaskedLM", "BertForSequenceClassification",
                        "BertForTokenClassification", "BertForQuestionAnswering"),
                List.of( ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "japanese_wordpiece_or_sentencepiece",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only_over_bert_backbone",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bert-japanese-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("vocab.txt"),
                        List.of("spiece.model"), List.of("tokenizer/tokenizer.json"),
                        List.of("tokenizer/vocab.txt"), List.of("tokenizer/spiece.model")),
                Map.of(
                        "word_tokenizer", "basic_mecab_sudachi_or_jumanpp",
                        "subword_tokenizer", "wordpiece_character_or_sentencepiece",
                        "status", "metadata_only_until_japanese_segmentation_runtime")));
    }
}
