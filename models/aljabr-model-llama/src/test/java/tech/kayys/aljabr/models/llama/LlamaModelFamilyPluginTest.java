package tech.kayys.tafkir.models.llama;

import org.junit.jupiter.api.Test;
import tech.kayys.tafkir.models.LLaMA3Family;
import tech.kayys.tafkir.models.LlamaFamily;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LlamaModelFamilyPluginTest {

    @Test
    void llamaPluginSatisfiesSharedModelFamilyContract() {
        LlamaModelFamilyPlugin plugin = new LlamaModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "llama model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void llamaFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new LlamaModelFamilyPlugin(),
                fixture("llama"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "llama fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesLlamaAndLlama3DirectArchitectureAdapters() {
        LlamaModelFamilyPlugin plugin = new LlamaModelFamilyPlugin();

        assertEquals(List.of("llama", "llama3"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .collect(Collectors.toList()));
        assertTrue(plugin.architectureAdapters().stream().anyMatch(adapter -> adapter instanceof LlamaFamily));
        assertTrue(plugin.architectureAdapters().stream().anyMatch(adapter -> adapter instanceof LLaMA3Family));
        assertTrue(plugin.descriptor().modelTypes().containsAll(List.of("llama", "llama3")));
    }

    @Test
    void llamaDirectWeightsUseHuggingFaceCausalLmLayout() {
        ModelArchitecture architecture = new LlamaFamily();

        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("lm_head.weight", architecture.lmHeadWeight());
        assertEquals("model.layers.4.self_attn.q_proj.weight", architecture.layerQueryWeight(4));
        assertEquals("model.layers.4.self_attn.k_proj.weight", architecture.layerKeyWeight(4));
        assertEquals("model.layers.4.self_attn.v_proj.weight", architecture.layerValueWeight(4));
        assertEquals("model.layers.4.self_attn.o_proj.weight", architecture.layerOutputWeight(4));
        assertEquals("model.layers.4.input_layernorm.weight", architecture.layerAttentionNormWeight(4));
        assertEquals("model.layers.4.post_attention_layernorm.weight", architecture.layerFfnNormWeight(4));
        assertEquals("model.layers.4.mlp.gate_proj.weight", architecture.layerFfnGateWeight(4));
        assertEquals("model.layers.4.mlp.up_proj.weight", architecture.layerFfnUpWeight(4));
        assertEquals("model.layers.4.mlp.down_proj.weight", architecture.layerFfnDownWeight(4));
    }

    @Test
    void llama3DirectWeightsStayCompatibleWithBaseLlamaLayout() {
        ModelArchitecture architecture = new LLaMA3Family();

        assertTrue(architecture.supportedArchClassNames().contains("LlamaForCausalLM"));
        assertEquals(List.of("llama", "llama3"), architecture.supportedModelTypes());
        assertEquals("model.layers.2.self_attn.q_proj.weight", architecture.layerQueryWeight(2));
        assertEquals("model.layers.2.mlp.gate_proj.weight", architecture.layerFfnGateWeight(2));
        assertEquals("model.layers.2.mlp.down_proj.weight", architecture.layerFfnDownWeight(2));
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                LlamaModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
