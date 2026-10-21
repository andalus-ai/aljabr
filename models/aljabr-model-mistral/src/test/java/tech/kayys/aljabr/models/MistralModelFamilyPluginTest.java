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

class MistralModelFamilyPluginTest {

    @Test
    void mistralPluginSatisfiesSharedModelFamilyContract() {
        MistralModelFamilyPlugin plugin = new MistralModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "mistral model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void mistralFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new MistralModelFamilyPlugin(),
                fixture("mistral"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "mistral fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesOnlyCoreMistralDirectArchitectureAdapter() {
        MistralModelFamilyPlugin plugin = new MistralModelFamilyPlugin();

        assertEquals(List.of("mistral"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof Mistral3Family);
        assertTrue(plugin.architectureAdapters().stream().noneMatch(adapter -> adapter instanceof MixtralFamily));
        assertEquals(List.of("mistral"), plugin.descriptor().modelTypes());
        assertEquals(List.of("MistralForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals(List.of("mistral-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void mistralDirectWeightsUseHuggingFaceCausalLmLayout() {
        Mistral3Family architecture = new Mistral3Family();

        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("lm_head.weight", architecture.lmHeadWeight());
        assertEquals("model.layers.8.self_attn.q_proj.weight", architecture.layerQueryWeight(8));
        assertEquals("model.layers.8.self_attn.k_proj.weight", architecture.layerKeyWeight(8));
        assertEquals("model.layers.8.self_attn.v_proj.weight", architecture.layerValueWeight(8));
        assertEquals("model.layers.8.self_attn.o_proj.weight", architecture.layerOutputWeight(8));
        assertEquals("model.layers.8.input_layernorm.weight", architecture.layerAttentionNormWeight(8));
        assertEquals("model.layers.8.post_attention_layernorm.weight", architecture.layerFfnNormWeight(8));
        assertEquals("model.layers.8.mlp.gate_proj.weight", architecture.layerFfnGateWeight(8));
        assertEquals("model.layers.8.mlp.up_proj.weight", architecture.layerFfnUpWeight(8));
        assertEquals("model.layers.8.mlp.down_proj.weight", architecture.layerFfnDownWeight(8));
    }

    @Test
    void mistralArchitectureClaimsSelectedCoreConfigFamily() {
        Mistral3Family architecture = new Mistral3Family();

        assertEquals(List.of("MistralForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("mistral"), architecture.supportedModelTypes());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                MistralModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
