package tech.kayys.tafkir.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CohereModelFamilyPluginTest {

    @Test
    void coherePluginSatisfiesSharedModelFamilyContract() {
        CohereModelFamilyPlugin plugin = new CohereModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "cohere model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void cohere2FixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new CohereModelFamilyPlugin(),
                fixture("cohere2"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "cohere2 fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesCohere2DirectArchitectureAdapterAndTokenizer() {
        CohereModelFamilyPlugin plugin = new CohereModelFamilyPlugin();

        assertEquals(List.of("cohere2"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof CohereR2Family);
        assertEquals(List.of("cohere", "cohere2"), plugin.descriptor().modelTypes());
        assertEquals(List.of("CohereForCausalLM", "Cohere2ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals("cohere2_text_adapter_only", plugin.descriptor().metadata().get("direct_safetensor_scope"));
        assertEquals(List.of("cohere-hf-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void cohere2DirectWeightsExposeTiedEmbeddingAndParallelSwiGluLayout() {
        CohereR2Family architecture = new CohereR2Family();

        assertEquals(List.of("Cohere2ForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("cohere2"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.embed_tokens.weight", architecture.lmHeadWeight());
        assertTrue(architecture.hasTiedEmbeddings());
        assertEquals("model.layers.12.self_attn.q_proj.weight", architecture.layerQueryWeight(12));
        assertEquals("model.layers.12.self_attn.k_proj.weight", architecture.layerKeyWeight(12));
        assertEquals("model.layers.12.self_attn.v_proj.weight", architecture.layerValueWeight(12));
        assertEquals("model.layers.12.self_attn.o_proj.weight", architecture.layerOutputWeight(12));
        assertEquals("model.layers.12.input_layernorm.weight", architecture.layerAttentionNormWeight(12));
        assertEquals("model.layers.12.input_layernorm.weight", architecture.layerFfnNormWeight(12));
        assertEquals("model.layers.12.mlp.gate_proj.weight", architecture.layerFfnGateWeight(12));
        assertEquals("model.layers.12.mlp.up_proj.weight", architecture.layerFfnUpWeight(12));
        assertEquals("model.layers.12.mlp.down_proj.weight", architecture.layerFfnDownWeight(12));
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                CohereModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
