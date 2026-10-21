package tech.kayys.aljabr.models.gemma3;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Gemma3ModelFamilyPluginTest {

    @Test
    void gemma3PluginSatisfiesSharedModelFamilyContract() {
        Gemma3ModelFamilyPlugin plugin = new Gemma3ModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma3 model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void gemma3TextFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma3ModelFamilyPlugin(),
                fixture("gemma3_text"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma3 text fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesGemma3TextDirectArchitectureAdapterAndTokenizer() {
        Gemma3ModelFamilyPlugin plugin = new Gemma3ModelFamilyPlugin();

        assertEquals(List.of("gemma3_text"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof Gemma3TextFamily);
        assertEquals(List.of("gemma3", "gemma3_text"), plugin.descriptor().modelTypes());
        assertEquals(List.of("Gemma3ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals("text_adapter_only", plugin.descriptor().metadata().get("direct_safetensor"));
        assertEquals("Gemma3ForConditionalGeneration", plugin.descriptor().metadata().get("multimodal_architecture"));
        assertEquals(List.of("gemma3-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void gemma3TextDirectWeightsExposeQkNormAndGemmaNormLayout() {
        Gemma3TextFamily architecture = new Gemma3TextFamily();

        assertEquals(List.of("Gemma3ForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("gemma3", "gemma3_text"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.embed_tokens.weight", architecture.lmHeadWeight());
        assertTrue(architecture.hasTiedEmbeddings());
        assertEquals("model.layers.5.self_attn.q_proj.weight", architecture.layerQueryWeight(5));
        assertEquals("model.layers.5.self_attn.k_proj.weight", architecture.layerKeyWeight(5));
        assertEquals("model.layers.5.self_attn.v_proj.weight", architecture.layerValueWeight(5));
        assertEquals("model.layers.5.self_attn.o_proj.weight", architecture.layerOutputWeight(5));
        assertEquals("model.layers.5.self_attn.q_norm.weight", architecture.layerQueryNormWeight(5));
        assertEquals("model.layers.5.self_attn.k_norm.weight", architecture.layerKeyNormWeight(5));
        assertEquals(architecture.layerQueryNormWeight(5), architecture.layerQNorm(5));
        assertEquals(architecture.layerKeyNormWeight(5), architecture.layerKNorm(5));
        assertEquals("model.layers.5.input_layernorm.weight", architecture.layerAttentionNormWeight(5));
        assertEquals("model.layers.5.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(5));
        assertEquals(architecture.layerPostAttnNormWeight(5), architecture.layerPostAttnNorm(5));
        assertEquals("model.layers.5.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(5));
        assertEquals("model.layers.5.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(5));
        assertEquals("model.layers.5.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(5));
    }

    @Test
    void gemma3TextRuntimeTraitsExposeTextAdapterBehavior() {
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        ModelRuntimeTraits traits = architecture.runtimeTraits(null);

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(architecture.usesNeoxRope());
        assertTrue(architecture.addOneToRmsNormWeight());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE, traits.promptBosPolicy());
        assertEquals(ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE,
                Gemma3RuntimeProfile.prompt().promptBosPolicy());
        assertFalse(traits.nativeBf16Matvec());
        assertTrue(traits.gemma3Text());
        assertFalse(traits.qwenText());
        assertFalse(traits.skipDefaultSystemPromptInjection());
        assertTrue(traits.attention().splitHalfRope());
        assertFalse(traits.attention().packedQkvProjection());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                Gemma3ModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
