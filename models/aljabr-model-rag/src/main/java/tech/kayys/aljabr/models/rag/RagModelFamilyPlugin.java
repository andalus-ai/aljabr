package tech.kayys.aljabr.models.rag;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RagModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "rag",
                "Retrieval-Augmented Generation",
                List.of("rag"),
                List.of("RagModel", "RagSequenceForGeneration", "RagTokenForGeneration"),
                List.of(
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                Map.of(
                        "bundle_profile", "optional",
                        "direct_safetensor", "ready_text_path_guarded_by_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<tech.kayys.aljabr.spi.model.ModelArchitecture> architectureAdapters() {
        return List.of(new RagFamily());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "rag-composite-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("question_encoder_tokenizer", "generator_tokenizer"),
                        List.of("tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_rag",
                        "retriever", "retrieval_rag",
                        "status", "optional")));
    }
}
