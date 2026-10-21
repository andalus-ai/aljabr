package tech.kayys.aljabr.models.prophetnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ProphetNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "prophetnet",
                "ProphetNet",
                List.of("prophetnet", "xlm-prophetnet", "xlm_prophetnet"),
                List.of("ProphetNetForConditionalGeneration", "ProphetNetForCausalLM",
                        "ProphetNetModel", "XLMProphetNetForConditionalGeneration"),
                List.of(
                        ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "wordpiece_prophetnet",
                        "direct_safetensor", "pending_prophetnet_n_stream_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "prophetnet-wordpiece",
                ModelTokenizerKind.WORD_PIECE,
                List.of(List.of("prophetnet.tokenizer"),
                        List.of("tokenizer/prophetnet.tokenizer"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of("pre_tokenizer", "bert-basic")));
    }
}
