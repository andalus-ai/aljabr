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

class YiModelFamilyPluginTest {

    @Test
    void yiPluginSatisfiesSharedModelFamilyContract() {
        YiModelFamilyPlugin plugin = new YiModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "yi model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void yiFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new YiModelFamilyPlugin(),
                fixture("yi"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "yi fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesYiDirectArchitectureAdapterAndTokenizer() {
        YiModelFamilyPlugin plugin = new YiModelFamilyPlugin();

        assertEquals(List.of("yi"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof YiFamily);
        assertEquals(List.of("yi", "yi_1_5"), plugin.descriptor().modelTypes());
        assertEquals(List.of("YiForCausalLM", "LlamaForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals("llama", plugin.descriptor().metadata().get("compatible_base"));
        assertEquals(List.of("yi-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void yiDirectWeightsUseLlamaCompatibleHuggingFaceLayout() {
        YiFamily architecture = new YiFamily();

        assertEquals(List.of("YiForCausalLM", "LlamaForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("yi", "yi_1_5"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("lm_head.weight", architecture.lmHeadWeight());
        assertEquals("model.layers.11.self_attn.q_proj.weight", architecture.layerQueryWeight(11));
        assertEquals("model.layers.11.self_attn.k_proj.weight", architecture.layerKeyWeight(11));
        assertEquals("model.layers.11.self_attn.v_proj.weight", architecture.layerValueWeight(11));
        assertEquals("model.layers.11.self_attn.o_proj.weight", architecture.layerOutputWeight(11));
        assertEquals("model.layers.11.input_layernorm.weight", architecture.layerAttentionNormWeight(11));
        assertEquals("model.layers.11.post_attention_layernorm.weight", architecture.layerFfnNormWeight(11));
        assertEquals("model.layers.11.mlp.gate_proj.weight", architecture.layerFfnGateWeight(11));
        assertEquals("model.layers.11.mlp.up_proj.weight", architecture.layerFfnUpWeight(11));
        assertEquals("model.layers.11.mlp.down_proj.weight", architecture.layerFfnDownWeight(11));
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                YiModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
