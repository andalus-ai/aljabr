package tech.kayys.aljabr.models.gemma2;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.FFNActivationType;
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

class Gemma2ModelFamilyPluginTest {

    @Test
    void gemma2PluginSatisfiesSharedModelFamilyContract() {
        Gemma2ModelFamilyPlugin plugin = new Gemma2ModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma2 model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void gemma2FixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma2ModelFamilyPlugin(),
                fixture("gemma2"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma2 fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesGemma2DirectArchitectureAdapterAndTokenizer() {
        Gemma2ModelFamilyPlugin plugin = new Gemma2ModelFamilyPlugin();

        assertEquals(List.of("gemma2"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof Gemma2Family);
        assertEquals(List.of("gemma2"), plugin.descriptor().modelTypes());
        assertEquals(List.of("Gemma2ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals(List.of("gemma2-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void gemma2DirectWeightsExposeTiedEmbeddingAndNormLayout() {
        Gemma2Family architecture = new Gemma2Family();

        assertEquals(List.of("Gemma2ForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("gemma2"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.embed_tokens.weight", architecture.lmHeadWeight());
        assertTrue(architecture.hasTiedEmbeddings());
        assertEquals("model.layers.7.self_attn.q_proj.weight", architecture.layerQueryWeight(7));
        assertEquals("model.layers.7.self_attn.k_proj.weight", architecture.layerKeyWeight(7));
        assertEquals("model.layers.7.self_attn.v_proj.weight", architecture.layerValueWeight(7));
        assertEquals("model.layers.7.self_attn.o_proj.weight", architecture.layerOutputWeight(7));
        assertEquals("model.layers.7.input_layernorm.weight", architecture.layerAttentionNormWeight(7));
        assertEquals("model.layers.7.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(7));
        assertEquals("model.layers.7.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(7));
        assertEquals("model.layers.7.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(7));
        assertEquals("model.layers.7.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(7));
        assertEquals("model.layers.7.mlp.gate_proj.weight", architecture.layerFfnGateWeight(7));
        assertEquals("model.layers.7.mlp.up_proj.weight", architecture.layerFfnUpWeight(7));
        assertEquals("model.layers.7.mlp.down_proj.weight", architecture.layerFfnDownWeight(7));
    }

    @Test
    void gemma2RuntimeTraitsExposeSoftCapsAndScaledEmbeddings() {
        Gemma2Family architecture = new Gemma2Family();

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(!architecture.usesNeoxRope());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(50.0f, architecture.defaultAttnSoftCap());
        assertEquals(30.0f, architecture.defaultFinalSoftCap());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                Gemma2ModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
