package tech.kayys.tafkir.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GemmaModelFamilyPluginTest {

    @Test
    void gemmaPluginSatisfiesSharedModelFamilyContract() {
        GemmaModelFamilyPlugin plugin = new GemmaModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "gemma model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void gemmaFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new GemmaModelFamilyPlugin(),
                fixture("gemma"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "gemma fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesGemmaDirectArchitectureAdapter() {
        GemmaModelFamilyPlugin plugin = new GemmaModelFamilyPlugin();

        assertEquals(List.of("gemma"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof GemmaFamily);
        assertEquals(List.of("gemma"), plugin.architectureAdapters().get(0).supportedModelTypes());
        assertEquals(List.of("GemmaForCausalLM"),
                plugin.architectureAdapters().get(0).supportedArchClassNames());
        assertTrue(plugin.descriptor().modelTypes().contains("gemma"));
        assertFalse(plugin.descriptor().modelTypes().contains("gemma4"));
        assertFalse(plugin.descriptor().modelTypes().contains("gemma4_text"));
        assertFalse(plugin.descriptor().modelTypes().contains("gemma4_unified"));
        assertFalse(plugin.descriptor().modelTypes().contains("gemma4_unified_text"));
        assertTrue(plugin.descriptor().architectureClassNames().contains("GemmaForCausalLM"));
        assertFalse(plugin.descriptor().architectureClassNames().contains("Gemma4ForCausalLM"));
        assertFalse(plugin.descriptor().architectureClassNames().contains("Gemma4ForConditionalGeneration"));
        assertFalse(plugin.descriptor().architectureClassNames().contains("Gemma4UnifiedForConditionalGeneration"));
        assertEquals("ready", plugin.descriptor().metadata().get("direct_safetensor"));
        assertFalse(plugin.descriptor().metadata().containsKey("unified_model_type"));
        assertTrue(plugin.unifiedRuntimeRequirements().isEmpty());
    }

    @Test
    void gemma4ConditionalGenerationIsOwnedByDedicatedGemma4Family() {
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(new GemmaModelFamilyPlugin());

        assertTrue(registry.resolve("gemma4", "Gemma4ForConditionalGeneration").notFound());
        assertEquals(List.of(), registry.architectureAdaptersFor(
                "gemma4",
                "Gemma4ForConditionalGeneration")
                .stream()
                .map(ModelArchitecture::id)
                .toList());
    }

    @Test
    void gemma4UnifiedConditionalGenerationIsOwnedByDedicatedGemma4Family() {
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(new GemmaModelFamilyPlugin());

        assertTrue(registry.resolve("gemma4_unified", "Gemma4UnifiedForConditionalGeneration").notFound());
        assertEquals(List.of(), registry.architectureAdaptersFor(
                "gemma4_unified",
                "Gemma4UnifiedForConditionalGeneration")
                .stream()
                .map(ModelArchitecture::id)
                .toList());
    }

    @Test
    void gemmaDirectWeightsExposeCoreAndGemmaSpecificLayout() {
        GemmaFamily architecture = new GemmaFamily();

        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.embed_tokens_per_layer.weight", architecture.embedTokensPerLayerWeight());
        assertEquals("model.per_layer_model_projection.weight", architecture.perLayerModelProjectionWeight());
        assertEquals("model.per_layer_projection_norm.weight", architecture.perLayerProjectionNormWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.layers.6.self_attn.q_proj.weight", architecture.layerQueryWeight(6));
        assertEquals("model.layers.6.self_attn.k_proj.weight", architecture.layerKeyWeight(6));
        assertEquals("model.layers.6.self_attn.v_proj.weight", architecture.layerValueWeight(6));
        assertEquals("model.layers.6.self_attn.o_proj.weight", architecture.layerOutputWeight(6));
        assertEquals("model.layers.6.input_layernorm.weight", architecture.layerAttentionNormWeight(6));
        assertEquals("model.layers.6.self_attn.q_norm.weight", architecture.layerQueryNormWeight(6));
        assertEquals("model.layers.6.self_attn.k_norm.weight", architecture.layerKeyNormWeight(6));
        assertEquals("model.layers.6.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(6));
        assertEquals("model.layers.6.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(6));
        assertEquals("model.layers.6.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(6));
        assertEquals("model.layers.6.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(6));
    }

    @Test
    void gemmaDirectWeightsExposeGemma4PerLayerInputs() {
        GemmaFamily architecture = new GemmaFamily();

        assertEquals("model.layers.3.per_layer_input_gate.weight", architecture.layerPerLayerInputGateWeight(3));
        assertEquals("model.layers.3.per_layer_projection.weight", architecture.layerPerLayerProjectionWeight(3));
        assertEquals("model.layers.3.post_per_layer_input_norm.weight",
                architecture.layerPostPerLayerInputNormWeight(3));
        assertEquals("model.layers.3.layer_scalar", architecture.layerScalarWeight(3));
    }

    @Test
    void gemmaRuntimeTraitsUseGeluNeoxRopeAndScaledEmbeddings() {
        GemmaFamily architecture = new GemmaFamily();
        ModelRuntimeTraits traits = architecture.runtimeTraits(null);

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(architecture.usesNeoxRope());
        assertTrue(architecture.addOneToRmsNormWeight());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(0.0f, architecture.defaultAttnSoftCap());
        assertEquals(0.0f, architecture.defaultFinalSoftCap());
        assertFalse(traits.nativeBf16Matvec());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE, traits.promptBosPolicy());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE,
                GemmaRuntimeProfile.prompt(null).promptBosPolicy());
        assertTrue(traits.allowedControlTokenTexts().isEmpty());
        assertFalse(traits.skipDefaultSystemPromptInjection());
    }

    @Test
    void gemmaRuntimeProfilePreservesGemma4CompatibilityPath() {
        GemmaFamily architecture = new GemmaFamily();
        ModelConfig config = ModelConfig.fromGgufMetadata(Map.of(
                "general.architecture", "gemma4_text",
                "gemma4_text.embedding_length_per_layer_input", 4096));

        ModelRuntimeTraits traits = architecture.runtimeTraits(config);

        assertTrue(traits.nativeBf16Matvec());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.NEVER, traits.promptBosPolicy());
        assertTrue(traits.skipDefaultSystemPromptInjection());
        assertTrue(traits.validateContinuationTokensByDecode());
        assertTrue(traits.rejectEmptyDecodedTokens());
        assertTrue(traits.allowedControlTokenTexts().contains("<|channel>"));
        assertTrue(traits.perLayerInputPath());
        assertTrue(traits.attention().splitHalfRope());
        assertTrue(traits.attention().preferNativeMetalBf16Linear());
        assertTrue(traits.attention().restrictLegacyMetalAttentionBridge());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                GemmaModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
