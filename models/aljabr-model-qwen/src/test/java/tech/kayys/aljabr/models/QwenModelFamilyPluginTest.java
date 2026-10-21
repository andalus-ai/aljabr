package tech.kayys.aljabr.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QwenModelFamilyPluginTest {

    @Test
    void qwenPluginSatisfiesSharedModelFamilyContract() {
        QwenModelFamilyPlugin plugin = new QwenModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "qwen model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void qwenFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new QwenModelFamilyPlugin(),
                fixture("qwen"));

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList(),
                "qwen fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesQwen2Qwen25AndQwen3DirectArchitectureAdapters() {
        QwenModelFamilyPlugin plugin = new QwenModelFamilyPlugin();

        assertEquals(List.of("qwen2", "qwen2.5", "qwen3"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().stream().anyMatch(adapter -> adapter instanceof Qwen2Family));
        assertTrue(plugin.architectureAdapters().stream().anyMatch(adapter -> adapter instanceof Qwen25Family));
        assertTrue(plugin.architectureAdapters().stream().anyMatch(adapter -> adapter instanceof Qwen3Family));
        assertEquals(List.of("qwen2", "qwen2.5", "qwen3"), plugin.descriptor().modelTypes());
        assertEquals(List.of("Qwen2ForCausalLM", "Qwen3ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals(List.of("qwen-hf-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void qwen2DirectWeightsExposeProjectionBiases() {
        Qwen2Family architecture = new Qwen2Family();

        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("lm_head.weight", architecture.lmHeadWeight());
        assertEquals("model.layers.9.self_attn.q_proj.weight", architecture.layerQueryWeight(9));
        assertEquals("model.layers.9.self_attn.k_proj.weight", architecture.layerKeyWeight(9));
        assertEquals("model.layers.9.self_attn.v_proj.weight", architecture.layerValueWeight(9));
        assertEquals("model.layers.9.self_attn.o_proj.weight", architecture.layerOutputWeight(9));
        assertEquals("model.layers.9.self_attn.q_proj.bias", architecture.layerQueryBias(9));
        assertEquals("model.layers.9.self_attn.k_proj.bias", architecture.layerKeyBias(9));
        assertEquals("model.layers.9.self_attn.v_proj.bias", architecture.layerValueBias(9));
        assertEquals("model.layers.9.self_attn.o_proj.bias", architecture.layerOutputBias(9));
        assertEquals("model.layers.9.mlp.gate_proj.bias", architecture.layerFfnGateBias(9));
        assertEquals("model.layers.9.mlp.up_proj.bias", architecture.layerFfnUpBias(9));
        assertEquals("model.layers.9.mlp.down_proj.bias", architecture.layerFfnDownBias(9));
    }

    @Test
    void qwen25DirectWeightsKeepQwen2LayoutAndAddQkNorms() {
        Qwen25Family architecture = new Qwen25Family();

        assertEquals(List.of("Qwen2ForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("qwen2.5"), architecture.supportedModelTypes());
        assertEquals("model.layers.4.self_attn.q_proj.weight", architecture.layerQueryWeight(4));
        assertEquals("model.layers.4.mlp.gate_proj.weight", architecture.layerFfnGateWeight(4));
        assertEquals("model.layers.4.mlp.up_proj.weight", architecture.layerFfnUpWeight(4));
        assertEquals("model.layers.4.mlp.down_proj.weight", architecture.layerFfnDownWeight(4));
        assertEquals("model.layers.4.self_attn.q_norm.weight", architecture.layerQueryNormWeight(4));
        assertEquals("model.layers.4.self_attn.k_norm.weight", architecture.layerKeyNormWeight(4));
    }

    @Test
    void qwenRuntimeProfileOwnsPromptDefaults() {
        ModelRuntimeTraits traits = QwenRuntimeProfile.text(new ModelConfig());

        assertTrue(traits.qwenText());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.DEFAULT, traits.promptBosPolicy());
        assertEquals(ModelRuntimeTraits.QWEN_DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        assertFalse(traits.skipDefaultSystemPromptInjection());
        assertFalse(traits.validateContinuationTokensByDecode());
        assertFalse(traits.rejectEmptyDecodedTokens());
        assertEquals(List.of(), traits.allowedControlTokenTexts().stream().toList());
        assertEquals(ModelRuntimeTraits.QWEN_DEFAULT_SYSTEM_PROMPT,
                QwenRuntimeProfile.prompt().defaultSystemPrompt());
    }

    @Test
    void qwenAdaptersUseSharedRuntimeProfileForCompactAttention() {
        ModelConfig compactConfig = ModelConfig.fromGgufMetadata(Map.of(
                "general.architecture", "qwen2",
                "qwen2.block_count", 24,
                "qwen2.embedding_length", 1536,
                "qwen2.feed_forward_length", 8960));

        for (ModelArchitecture adapter : List.of(new Qwen2Family(), new Qwen25Family(), new Qwen3Family())) {
            ModelRuntimeTraits traits = adapter.runtimeTraits(compactConfig);

            assertTrue(traits.qwenText(), adapter.id());
            assertTrue(traits.attention().compactAttentionMatvecCandidate(), adapter.id());
            assertEquals(
                    ModelAttentionTraitsPolicy.DEFAULT_QWEN_PAGED_METAL_PREFILL_MAX_TOKENS,
                    traits.attention().defaultPagedMetalPrefillMaxTokens(),
                    adapter.id());
        }
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                QwenModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
