package tech.kayys.tafkir.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PhiModelFamilyPluginTest {

    @Test
    void phiPluginSatisfiesSharedModelFamilyContract() {
        PhiModelFamilyPlugin plugin = new PhiModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "phi model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void phiFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new PhiModelFamilyPlugin(),
                fixture("phi"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "phi fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void phi4MiniFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new PhiModelFamilyPlugin(),
                fixture("phi4-mini"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "phi4-mini fixture should stay covered by the Phi model-family adapter");
    }

    @Test
    void publishesPhiDirectArchitectureAdapterAndTokenizers() {
        PhiModelFamilyPlugin plugin = new PhiModelFamilyPlugin();

        assertEquals(List.of("phi"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof PhiFamily);
        assertTrue(plugin.descriptor().modelTypes().containsAll(List.of("phi", "phi3", "phi4")));
        assertTrue(plugin.descriptor().architectureClassNames()
                .containsAll(List.of("PhiForCausalLM", "Phi3ForCausalLM", "Phi3SmallForCausalLM",
                        "Phi4ForCausalLM")));
        assertEquals(List.of("phi-spm-bpe", "phi-hf-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void phiDirectWeightsExposeFusedQkvAndFusedGateUpLayout() {
        PhiFamily architecture = new PhiFamily();

        assertTrue(architecture.hasFusedQKV());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals(List.of("model.norm.weight", "model.final_layernorm.weight"),
                architecture.finalNormWeightCandidates());
        assertNull(architecture.finalNormBias());
        assertEquals("model.layers.5.self_attn.qkv_proj.weight", architecture.layerFusedQKVWeight(5));
        assertEquals(architecture.layerFusedQKVWeight(5), architecture.layerQueryWeight(5));
        assertEquals(architecture.layerFusedQKVWeight(5), architecture.layerKeyWeight(5));
        assertEquals(architecture.layerFusedQKVWeight(5), architecture.layerValueWeight(5));
        assertEquals("model.layers.5.self_attn.o_proj.weight", architecture.layerOutputWeight(5));
        assertEquals("model.layers.5.input_layernorm.weight", architecture.layerAttentionNormWeight(5));
        assertEquals("model.layers.5.mlp.gate_up_proj.weight", architecture.layerFfnGateWeight(5));
        assertEquals("model.layers.5.mlp.gate_up_proj.weight", architecture.layerFfnUpWeight(5));
        assertEquals("model.layers.5.mlp.down_proj.weight", architecture.layerFfnDownWeight(5));
        assertEquals("model.layers.5.post_attention_layernorm.weight", architecture.layerFfnNormWeight(5));
    }

    @Test
    void phiArchitectureClaimsPhiAndPhi3ConfigFamilies() {
        PhiFamily architecture = new PhiFamily();

        assertEquals(List.of("PhiForCausalLM", "Phi3ForCausalLM", "Phi3SmallForCausalLM", "Phi4ForCausalLM"),
                architecture.supportedArchClassNames());
        assertEquals(List.of("phi", "phi3", "phi4"), architecture.supportedModelTypes());
        assertTrue(architecture.usesRmsNorm());
    }

    @Test
    void phiRuntimeTraitsAdvertisePackedQkvProjectionForPhiSizedConfig() throws Exception {
        PhiFamily architecture = new PhiFamily();
        ModelConfig config = phiSizedConfig();

        ModelRuntimeTraits traits = architecture.runtimeTraits(config);

        assertFalse(traits.nativeBf16Matvec());
        assertFalse(traits.gemma3Text());
        assertFalse(traits.qwenText());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.DEFAULT, traits.promptBosPolicy());
        assertEquals(ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        assertEquals(ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT, PhiRuntimeProfile.prompt().defaultSystemPrompt());
        assertTrue(traits.attention().packedQkvProjection());
        assertTrue(traits.attention().largeAttentionMatvecCandidate());
    }

    @Test
    void phi4MiniRuntimeTraitsPreservePackedGqaQkvLayout() throws Exception {
        PhiFamily architecture = new PhiFamily();
        ModelConfig config = phi4MiniSizedConfig();

        ModelRuntimeTraits traits = architecture.runtimeTraits(config);

        assertEquals(3072, config.hiddenSize());
        assertEquals(24, config.numAttentionHeads());
        assertEquals(8, config.resolvedNumKvHeads());
        assertEquals(128, config.resolvedHeadDim());
        assertEquals(5120, packedQkvProjectionDim(config));
        assertTrue(traits.attention().packedQkvProjection());
    }

    private static ModelConfig phiSizedConfig() throws Exception {
        ModelConfig config = new ModelConfig();
        config.overrideHiddenSize(3072);
        config.overrideNumHiddenLayers(32);
        config.overrideIntermediateSize(8192);
        config.overrideNumAttentionHeads(32);
        config.overrideNumKeyValueHeads(32);
        config.overrideHeadDim(96);
        return config;
    }

    private static ModelConfig phi4MiniSizedConfig() {
        ModelConfig config = new ModelConfig();
        config.overrideHiddenSize(3072);
        config.overrideNumHiddenLayers(32);
        config.overrideIntermediateSize(8192);
        config.overrideNumAttentionHeads(24);
        config.overrideNumKeyValueHeads(8);
        return config;
    }

    private static int packedQkvProjectionDim(ModelConfig config) {
        return (config.numAttentionHeads() + 2 * config.resolvedNumKvHeads()) * config.resolvedHeadDim();
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                PhiModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
