package tech.kayys.aljabr.models.gptneoxjapanese;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class GptNeoXJapaneseModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gpt_neox_japanese",
                "GPT-NeoX Japanese",
                List.of("gpt_neox_japanese", "gpt-neox-japanese", "gptneox_japanese"),
                List.of("GPTNeoXJapaneseForCausalLM", "GPTNeoXJapaneseModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "gpt_neox_japanese_subword_bpe",
                        "direct_safetensor", "pending_gpt_neox_japanese_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "gpt-neox-japanese-subword-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.txt", "emoji.json"),
                        List.of("tokenizer/vocab.txt", "tokenizer/emoji.json")),
                Map.of(
                        "pre_tokenizer", "japanese-subword-bpe",
                        "status", "metadata_only_until_gpt_neox_japanese_tokenizer_runtime")));
    }
}
