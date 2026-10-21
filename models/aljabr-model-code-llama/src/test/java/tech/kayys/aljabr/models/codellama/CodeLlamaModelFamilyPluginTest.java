package tech.kayys.aljabr.models.codellama;

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

class CodeLlamaModelFamilyPluginTest {

    @Test
    void codeLlamaPluginSatisfiesSharedModelFamilyContract() {
        CodeLlamaModelFamilyPlugin plugin = new CodeLlamaModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "code llama model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void codeLlamaFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new CodeLlamaModelFamilyPlugin(),
                fixture("code_llama"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "code llama fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesCodeLlamaDirectArchitectureAdapterAndTokenizer() {
        CodeLlamaModelFamilyPlugin plugin = new CodeLlamaModelFamilyPlugin();

        assertEquals(List.of("code_llama"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof CodeLlamaFamily);
        assertEquals(List.of("code_llama"), plugin.descriptor().modelTypes());
        assertEquals(List.of("LlamaForCausalLM", "CodeLlamaTokenizer"), plugin.descriptor().architectureClassNames());
        assertEquals("3rdparty/transformers/src/transformers/models/llama",
                plugin.descriptor().metadata().get("modeling_origin"));
        assertEquals(List.of("code-llama-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void codeLlamaDirectWeightsUseLlamaCompatibleHuggingFaceLayout() {
        CodeLlamaFamily architecture = new CodeLlamaFamily();

        assertEquals(List.of("LlamaForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("code_llama"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("lm_head.weight", architecture.lmHeadWeight());
        assertEquals("model.layers.10.self_attn.q_proj.weight", architecture.layerQueryWeight(10));
        assertEquals("model.layers.10.self_attn.k_proj.weight", architecture.layerKeyWeight(10));
        assertEquals("model.layers.10.self_attn.v_proj.weight", architecture.layerValueWeight(10));
        assertEquals("model.layers.10.self_attn.o_proj.weight", architecture.layerOutputWeight(10));
        assertEquals("model.layers.10.input_layernorm.weight", architecture.layerAttentionNormWeight(10));
        assertEquals("model.layers.10.post_attention_layernorm.weight", architecture.layerFfnNormWeight(10));
        assertEquals("model.layers.10.mlp.gate_proj.weight", architecture.layerFfnGateWeight(10));
        assertEquals("model.layers.10.mlp.up_proj.weight", architecture.layerFfnUpWeight(10));
        assertEquals("model.layers.10.mlp.down_proj.weight", architecture.layerFfnDownWeight(10));
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                CodeLlamaModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
