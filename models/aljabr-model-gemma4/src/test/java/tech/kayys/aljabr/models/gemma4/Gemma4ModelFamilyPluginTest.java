package tech.kayys.aljabr.models.gemma4;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.FFNActivationType;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDirectSupport;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.aljabr.spi.model.ModelFamilyProblemCodes;
import tech.kayys.aljabr.spi.model.ModelFamilyResolution;
import tech.kayys.aljabr.spi.model.ModelFamilyRuntimeCompatibility;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Gemma4ModelFamilyPluginTest {

    @Test
    void gemma4PluginSatisfiesSharedModelFamilyContract() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void gemma4FixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma4ModelFamilyPlugin(),
                fixture("gemma4"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 fixture should match descriptor, tokenizer, and guarded direct adapter claims");
    }

    @Test
    void gemma4Unified12bFixtureMatchesDescriptorTokenizerAndTextAdapter() throws Exception {
        Path fixture = fixture("gemma4-12b-it");
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma4ModelFamilyPlugin(),
                fixture);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 unified fixture should match descriptor, tokenizer, and nested text adapter claims");
        assertTrue(fixtureConfig(fixture).contains("\"Gemma4UnifiedForConditionalGeneration\""));
    }

    @Test
    void gemma4QatQ40FixtureMatchesDescriptorAndPinsQuantizationShape() throws Exception {
        Path fixture = fixture("gemma4-12b-it-qat-q4_0");
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma4ModelFamilyPlugin(),
                fixture);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 QAT Q4_0 fixture should preserve descriptor, tokenizer, and text adapter claims");

        String config = fixtureConfig(fixture);
        assertTrue(config.contains("\"_name_or_path\": \"google/gemma-4-12B-it-qat-q4_0-gguf\""));
        assertTrue(config.contains("\"family\": \"gemma4_qat\""));
        assertTrue(config.contains("\"format\": \"q4_0\""));
        assertTrue(config.contains("\"container\": \"gguf\""));
        assertTrue(config.contains("\"loader_scope\": \"metadata_only_pending_q4_0_weight_loader\""));
    }

    @Test
    void gemma4QatMobileFixtureMatchesDescriptorAndPinsMobileQuantizationShape() throws Exception {
        Path fixture = fixture("gemma4-e2b-it-qat-mobile");
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma4ModelFamilyPlugin(),
                fixture);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 mobile QAT fixture should preserve descriptor, tokenizer, and text adapter claims");

        String config = fixtureConfig(fixture);
        assertTrue(config.contains("\"_name_or_path\": \"google/gemma-4-E2B-it-qat-mobile-transformers\""));
        assertTrue(config.contains("\"family\": \"gemma4_qat\""));
        assertTrue(config.contains("\"format\": \"mobile\""));
        assertTrue(config.contains("\"container\": \"transformers\""));
        assertTrue(config.contains("\"activation_scaling\": \"static\""));
        assertTrue(config.contains("\"channel_wise\": true"));
        assertTrue(config.contains("\"token_generation_bits\": 2"));
        assertTrue(config.contains("\"loader_scope\": \"metadata_only_pending_mobile_quant_loader\""));
    }

    @Test
    void publishesGemma4TextDirectArchitectureAdapterAndTokenizer() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();

        assertEquals(List.of("gemma4"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof Gemma4Family);
        assertEquals(List.of("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio",
                        "gemma4_unified", "gemma4_unified_text"),
                plugin.descriptor().modelTypes());
        assertTrue(plugin.descriptor().architectureClassNames().contains("Gemma4ForMultimodalLM"));
        assertTrue(plugin.descriptor().architectureClassNames().contains("Gemma4UnifiedForConditionalGeneration"));
        assertTrue(plugin.descriptor().architectureClassNames().contains("Gemma4ForImageTextToText"));
        assertTrue(plugin.descriptor().capabilities().contains(ModelFamilyCapability.MULTIMODAL));
        assertTrue(plugin.descriptor().capabilities().contains(ModelFamilyCapability.VISION));
        assertTrue(plugin.descriptor().capabilities().contains(ModelFamilyCapability.AUDIO));
        assertTrue(plugin.descriptor().capabilities().contains(ModelFamilyCapability.MOE));
        assertEquals("ready_text_path_guarded_by_runtime",
                plugin.descriptor().metadata().get("direct_safetensor"));
        assertEquals("text_only_gemma4_text_and_unified_text",
                plugin.descriptor().metadata().get("direct_safetensor_scope"));
        assertEquals("pending_audio_vision_video_embedder_runtime",
                plugin.descriptor().metadata().get("multimodal_direct_safetensor"));
        assertEquals("pending_packed_expert_router_runtime",
                plugin.descriptor().metadata().get("moe_direct_safetensor"));
        assertEquals("gemma4_unified", plugin.descriptor().metadata().get("unified_model_type"));
        assertEquals(1, plugin.unifiedRuntimeRequirements().size());
        assertEquals("gemma4_unified", plugin.unifiedRuntimeRequirements().getFirst().modelType());
        assertEquals(List.of("text", "image", "audio", "video"),
                plugin.unifiedRuntimeRequirements().getFirst().requiredInputModalities());
        assertTrue(plugin.unifiedRuntimeRequirements().getFirst().productionReadyRequired());
        assertEquals("google/gemma-4-12B-it", plugin.descriptor().metadata().get("checkpoint_gemma_4_12b_it"));
        assertEquals("dense_unified_encoder_free_multimodal",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_12b_it_architecture"));
        assertEquals("dedicated_gpu_laptop_or_unified_memory_16gb",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_12b_it_local_target"));
        assertEquals("companion_multi_token_prediction_model_available",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_12b_it_mtp"));
        assertEquals("litert-community/gemma-4-12B-it-litert-lm",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_12b_it_litert_lm"));
        assertEquals("metadata_only_pending_litert_lm_runner",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_12b_it_litert_lm_scope"));
        assertEquals("google/gemma-4-qat-q4-0",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_qat_q4_0_collection"));
        assertEquals("unquantized,gguf,compressed_tensors_w4a16",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_qat_q4_0_formats"));
        assertEquals("metadata_only_pending_q4_0_weight_loader",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_qat_q4_0_scope"));
        assertEquals("google/gemma-4-qat-mobile",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_qat_mobile_collection"));
        assertEquals("transformers,compressed_tensors_mobile",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_qat_mobile_formats"));
        assertEquals("metadata_only_pending_mobile_quant_loader",
                plugin.descriptor().metadata().get("checkpoint_gemma_4_qat_mobile_scope"));
        assertEquals("google/gemma-4-qat-q4-0",
                plugin.unifiedRuntimeRequirements().getFirst().metadata().get("qat_q4_0_collection"));
        assertEquals("google/gemma-4-qat-mobile",
                plugin.unifiedRuntimeRequirements().getFirst().metadata().get("qat_mobile_collection"));
        assertEquals("dense_unified_encoder_free_multimodal",
                plugin.unifiedRuntimeRequirements().getFirst().metadata().get("architecture"));
        assertEquals("litert-community/gemma-4-12B-it-litert-lm",
                plugin.unifiedRuntimeRequirements().getFirst().metadata().get("litert_lm_repo"));
        assertEquals(List.of("gemma4-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void gemma4DirectWeightsExposePerLayerInputsAndQkNormLayout() {
        Gemma4Family architecture = new Gemma4Family();

        assertEquals(List.of("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
                        "Gemma4ForImageTextToText", "Gemma4ForMultimodalLM",
                        "Gemma4UnifiedForConditionalGeneration"),
                architecture.supportedArchClassNames());
        assertEquals(List.of("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"),
                architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.embed_tokens_per_layer.weight", architecture.embedTokensPerLayerWeight());
        assertEquals("model.per_layer_model_projection.weight", architecture.perLayerModelProjectionWeight());
        assertEquals("model.per_layer_projection_norm.weight", architecture.perLayerProjectionNormWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.layers.4.self_attn.q_proj.weight", architecture.layerQueryWeight(4));
        assertEquals("model.layers.4.self_attn.k_proj.weight", architecture.layerKeyWeight(4));
        assertEquals("model.layers.4.self_attn.v_proj.weight", architecture.layerValueWeight(4));
        assertEquals("model.layers.4.self_attn.o_proj.weight", architecture.layerOutputWeight(4));
        assertEquals("model.layers.4.self_attn.q_norm.weight", architecture.layerQueryNormWeight(4));
        assertEquals("model.layers.4.self_attn.k_norm.weight", architecture.layerKeyNormWeight(4));
        assertEquals("model.layers.4.input_layernorm.weight", architecture.layerAttentionNormWeight(4));
        assertEquals("model.layers.4.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(4));
        assertEquals("model.layers.4.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(4));
        assertEquals("model.layers.4.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(4));
        assertEquals("model.layers.4.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(4));
        assertEquals("model.layers.4.per_layer_input_gate.weight", architecture.layerPerLayerInputGateWeight(4));
        assertEquals("model.layers.4.per_layer_projection.weight", architecture.layerPerLayerProjectionWeight(4));
        assertEquals("model.layers.4.post_per_layer_input_norm.weight",
                architecture.layerPostPerLayerInputNormWeight(4));
        assertEquals("model.layers.4.layer_scalar", architecture.layerScalarWeight(4));
    }

    @Test
    void gemma4DirectWeightsExposeUnifiedTextWrapperCandidates() {
        Gemma4Family architecture = new Gemma4Family();

        assertEquals(List.of("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
                architecture.embedTokensWeightCandidates());
        assertEquals(List.of("model.norm.weight", "model.language_model.norm.weight"),
                architecture.finalNormWeightCandidates());
        assertEquals(List.of("lm_head.weight", "model.language_model.lm_head.weight",
                        "language_model.lm_head.weight"),
                architecture.lmHeadWeightCandidates());
        assertEquals(List.of("model.layers.4.self_attn.q_proj.weight",
                        "model.language_model.layers.4.self_attn.q_proj.weight"),
                architecture.layerQueryWeightCandidates(4));
        assertEquals(List.of("model.layers.4.self_attn.k_proj.weight",
                        "model.language_model.layers.4.self_attn.k_proj.weight"),
                architecture.layerKeyWeightCandidates(4));
        assertEquals(List.of("model.layers.4.self_attn.v_proj.weight",
                        "model.language_model.layers.4.self_attn.v_proj.weight"),
                architecture.layerValueWeightCandidates(4));
        assertEquals(List.of("model.layers.4.self_attn.o_proj.weight",
                        "model.language_model.layers.4.self_attn.o_proj.weight"),
                architecture.layerOutputWeightCandidates(4));
        assertEquals(List.of("model.layers.4.input_layernorm.weight",
                        "model.language_model.layers.4.input_layernorm.weight"),
                architecture.layerAttentionNormWeightCandidates(4));
        assertEquals(List.of("model.layers.4.self_attn.q_norm.weight",
                        "model.language_model.layers.4.self_attn.q_norm.weight"),
                architecture.layerQueryNormWeightCandidates(4));
        assertEquals(List.of("model.layers.4.self_attn.k_norm.weight",
                        "model.language_model.layers.4.self_attn.k_norm.weight"),
                architecture.layerKeyNormWeightCandidates(4));
        assertEquals(List.of("model.layers.4.mlp.gate_proj.weight",
                        "model.language_model.layers.4.mlp.gate_proj.weight"),
                architecture.layerFfnGateWeightCandidates(4));
        assertEquals(List.of("model.layers.4.mlp.up_proj.weight",
                        "model.language_model.layers.4.mlp.up_proj.weight"),
                architecture.layerFfnUpWeightCandidates(4));
        assertEquals(List.of("model.layers.4.mlp.down_proj.weight",
                        "model.language_model.layers.4.mlp.down_proj.weight"),
                architecture.layerFfnDownWeightCandidates(4));
        assertEquals(List.of("model.layers.4.post_feedforward_layernorm.weight",
                        "model.language_model.layers.4.post_feedforward_layernorm.weight"),
                architecture.layerPostFfnNormWeightCandidates(4));
        assertEquals(List.of("model.layers.4.layer_scalar", "model.language_model.layers.4.layer_scalar"),
                architecture.layerScalarWeightCandidates(4));
    }

    @Test
    void gemma4DirectWeightsExposeUnifiedPackedMoeCandidates() {
        Gemma4Family architecture = new Gemma4Family();

        assertEquals(List.of("model.layers.4.mlp.router.weight",
                        "model.language_model.layers.4.mlp.router.weight"),
                architecture.layerMoeGateWeightCandidates(4));
        assertEquals(List.of("model.layers.4.mlp.experts.7.gate_proj.weight",
                        "model.language_model.layers.4.mlp.experts.7.gate_proj.weight"),
                architecture.expertGateWeightCandidates(4, 7));
        assertEquals(List.of("model.layers.4.mlp.experts.7.up_proj.weight",
                        "model.language_model.layers.4.mlp.experts.7.up_proj.weight"),
                architecture.expertUpWeightCandidates(4, 7));
        assertEquals(List.of("model.layers.4.mlp.experts.7.down_proj.weight",
                        "model.language_model.layers.4.mlp.experts.7.down_proj.weight"),
                architecture.expertDownWeightCandidates(4, 7));
    }

    @Test
    void gemma4RuntimeTraitsExposeTextRuntimePoliciesWithoutConfig() {
        Gemma4Family architecture = new Gemma4Family();
        ModelRuntimeTraits traits = architecture.runtimeTraits(null);

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(architecture.usesNeoxRope());
        assertTrue(architecture.addOneToRmsNormWeight());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(0.0f, architecture.defaultAttnSoftCap());
        assertEquals(0.0f, architecture.defaultFinalSoftCap());
        assertTrue(!traits.nativeBf16Matvec());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE, traits.promptBosPolicy());
        assertTrue(traits.allowedControlTokenTexts().isEmpty());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.TURN_AWARE,
                Gemma4RuntimeProfile.prompt(null).promptBosPolicy());
    }

    @Test
    void gemma4RuntimeTraitsExposeTextConfigPolicies() {
        Gemma4Family architecture = new Gemma4Family();
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
        assertTrue(traits.attention().supportsForcedDenseAttention());
    }

    @Test
    void gemma4UnifiedRuntimeTraitsUseGemma4TextPolicies() {
        Gemma4Family architecture = new Gemma4Family();
        ModelConfig config = ModelConfig.fromGgufMetadata(Map.of(
                "general.architecture", "gemma4_unified",
                "gemma4_unified.embedding_length_per_layer_input", 5120));

        ModelRuntimeTraits traits = architecture.runtimeTraits(config);

        assertEquals("gemma4_unified", config.modelType());
        assertEquals("Gemma4ForMultimodalLM", config.primaryArchitecture());
        assertTrue(traits.nativeBf16Matvec());
        assertTrue(traits.skipDefaultSystemPromptInjection());
        assertTrue(traits.attention().splitHalfRope());
    }

    @Test
    void gemma4UnifiedVocabOnlyPerLayerMetadataDoesNotEnablePerLayerInputPath() {
        Gemma4Family architecture = new Gemma4Family();
        ModelConfig config = ModelConfig.fromGgufMetadata(Map.of(
                "general.architecture", "gemma4_unified",
                "gemma4_unified.vocab_size_per_layer_input", 262144));

        ModelRuntimeTraits traits = architecture.runtimeTraits(config);

        assertTrue(traits.nativeBf16Matvec());
        assertFalse(traits.perLayerInputPath());
    }

    @Test
    void gemma4UnifiedResolvesTextOnlyDirectSafetensorCompatibility() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(plugin);

        ModelFamilyResolution resolution = registry.resolve("gemma4_unified", "Gemma4ForMultimodalLM");
        ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(resolution);

        assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
        assertEquals(List.of("gemma4"), resolution.familyIds());
        assertEquals(List.of("gemma4-spm-bpe"), resolution.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
        assertEquals(ModelFamilyDirectSupport.READY,
                resolution.primarySupportReport().orElseThrow().directSafetensorStatus());
        assertTrue(resolution.primaryRuntimeManifest().orElseThrow().modelTypes().contains("gemma4_unified"));
        assertEquals("gemma4_unified",
                resolution.primaryRuntimeManifest().orElseThrow().unifiedRuntimeRequirements().getFirst().modelType());
        assertTrue(resolution.primaryRuntimeManifest().orElseThrow().requiresUnifiedRuntime());
        assertTrue(compatibility.compatible());
        assertEquals(List.of("gemma4"), compatibility.architectureAdapterIds());
        assertEquals("gemma4", compatibility.selectedArchitectureAdapterId());
        assertTrue(compatibility.problemCodes().stream().noneMatch("model_family_direct_safetensor_not_ready"::equals));
        assertTrue(compatibility.problemCodes().stream().noneMatch("model_family_architecture_adapter_unmatched"::equals));
        assertTrue(compatibility.problemCodes().stream().noneMatch("model_family_not_found"::equals));
    }

    @Test
    void gemma4UnifiedConditionalGenerationResolvesDirectSafetensorAdapter() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(plugin);

        ModelFamilyResolution resolution =
                registry.resolve("gemma4_unified", "Gemma4UnifiedForConditionalGeneration");
        ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(resolution);

        assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
        assertEquals(List.of("gemma4"), resolution.familyIds());
        assertEquals(ModelFamilyDirectSupport.READY,
                resolution.primarySupportReport().orElseThrow().directSafetensorStatus());
        assertTrue(compatibility.compatible());
        assertEquals(List.of("gemma4"), compatibility.architectureAdapterIds());
        assertEquals("gemma4", compatibility.selectedArchitectureAdapterId());
        assertTrue(compatibility.problemCodes().stream().noneMatch("model_family_not_found"::equals));
        assertTrue(compatibility.problemCodes().stream()
                .noneMatch("model_family_architecture_adapter_unmatched"::equals));
    }

    @Test
    void gemma4ImageTextToTextResolvesDirectSafetensorAdapter() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(plugin);

        ModelFamilyResolution resolution = registry.resolve("gemma4_unified", "Gemma4ForImageTextToText");
        ModelFamilyRuntimeCompatibility compatibility = registry.directSafetensorCompatibility(resolution);

        assertEquals(ModelFamilyResolution.Status.RESOLVED, resolution.status());
        assertEquals(List.of("gemma4"), resolution.familyIds());
        assertEquals(ModelFamilyDirectSupport.READY,
                resolution.primarySupportReport().orElseThrow().directSafetensorStatus());
        assertTrue(compatibility.compatible());
        assertEquals(List.of("gemma4"), compatibility.architectureAdapterIds());
        assertEquals("gemma4", compatibility.selectedArchitectureAdapterId());
        assertTrue(compatibility.problemCodes().stream().noneMatch("model_family_not_found"::equals));
        assertTrue(compatibility.problemCodes().stream()
                .noneMatch("model_family_architecture_adapter_unmatched"::equals));
    }

    @Test
    void gemma4QatQ40DirectCompatibilityReportsPendingQuantizedLoader() throws Exception {
        ModelFamilyRuntimeCompatibility compatibility = directCompatibilityForFixture("gemma4-12b-it-qat-q4_0");

        assertTrue(!compatibility.compatible());
        assertTrue(compatibility.problemCodes().contains(
                ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING));
        assertTrue(compatibility.problemCodes().contains(ModelFamilyProblemCodes.QAT_Q4_0_LOADER_PENDING));
        assertTrue(compatibility.problemCodes().stream()
                .noneMatch(ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING::equals));
        assertTrue(compatibility.remediationHints().stream()
                .anyMatch(hint -> hint.contains("q4_0 quantized weights in gguf")));
    }

    @Test
    void gemma4QatMobileDirectCompatibilityReportsPendingMobileLoader() throws Exception {
        ModelFamilyRuntimeCompatibility compatibility = directCompatibilityForFixture("gemma4-e2b-it-qat-mobile");

        assertTrue(!compatibility.compatible());
        assertTrue(compatibility.problemCodes().contains(
                ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING));
        assertTrue(compatibility.problemCodes().contains(ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING));
        assertTrue(compatibility.problemCodes().stream()
                .noneMatch(ModelFamilyProblemCodes.QAT_Q4_0_LOADER_PENDING::equals));
        assertTrue(compatibility.remediationHints().stream()
                .anyMatch(hint -> hint.contains("mobile quantized weights in transformers")));
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                Gemma4ModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }

    private static String fixtureConfig(Path fixture) throws Exception {
        return Files.readString(fixture.resolve("config.json"));
    }

    private static ModelFamilyRuntimeCompatibility directCompatibilityForFixture(String familyId) throws Exception {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        registry.register(plugin);
        ModelFamilyResolution resolution = registry.resolve("gemma4_unified", "Gemma4ForMultimodalLM");
        return registry.directSafetensorCompatibility(resolution, fixture(familyId));
    }
}
