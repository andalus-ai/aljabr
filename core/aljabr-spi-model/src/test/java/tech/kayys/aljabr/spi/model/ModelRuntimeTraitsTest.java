/*
 * Aljabr Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelRuntimeTraitsTest {

        @Test
        void genericTextUsesGenericDefaultSystemPrompt() {
                ModelRuntimeTraits traits = ModelRuntimeTraits.builder().build();

                assertFalse(traits.skipDefaultSystemPromptInjection());
                assertEquals(ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        }

        @Test
        void qwenTextUsesQwenDefaultSystemPrompt() {
                ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                                .qwenText()
                                .build();

                assertFalse(traits.skipDefaultSystemPromptInjection());
                assertEquals(ModelRuntimeTraits.QWEN_DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        }

        @Test
        void nativeBf16MatvecSkipsDefaultSystemPromptInjection() {
                ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                                .nativeBf16Matvec()
                                .build();

                assertTrue(traits.skipDefaultSystemPromptInjection());
        }

        @Test
        void builderDerivesPromptAndAttentionDefaultsFromNamedFlags() {
                ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                                .nativeBf16Matvec()
                                .perLayerInputPath()
                                .build();

                assertTrue(traits.nativeBf16Matvec());
                assertTrue(traits.perLayerInputPath());
                assertEquals(ModelRuntimeTraits.PromptBosPolicy.NEVER, traits.promptBosPolicy());
                assertTrue(traits.allowedControlTokenTexts().contains("<|channel>"));
                assertTrue(traits.validateContinuationTokensByDecode());
                assertTrue(traits.rejectEmptyDecodedTokens());
                assertTrue(traits.attention().splitHalfRope());
                assertTrue(traits.attention().restrictLegacyMetalAttentionBridge());
        }

        @Test
        void builderCanCopyExistingTraitsAndAddDetectedModalities() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"qwen2_vl","architectures":["Qwen2VLForConditionalGeneration"]}
                                """, ModelConfig.class);
                ModelRuntimeTraits base = ModelRuntimeTraits.builder()
                                .qwenText()
                                .build();

                ModelRuntimeTraits traits = ModelRuntimeTraits.builder(base)
                                .modalities(ModelModalityTraits.fromConfig(config))
                                .build();

                assertTrue(traits.qwenText());
                assertEquals(ModelRuntimeTraits.QWEN_DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
                assertFalse(traits.audioModel());
                assertTrue(traits.visionModel());
                assertTrue(traits.multimodalModel());
        }

        @Test
        void whisperConfigIsAudioModel() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}
                                """, ModelConfig.class);

                assertTrue(ModelRuntimeTraits.fallbackFromConfig(config).audioModel());
                assertFalse(ModelRuntimeTraits.fallbackFromConfig(config).visionModel());
                assertTrue(ModelRuntimeTraits.fallbackFromConfig(config).multimodalModel());
        }

        @Test
        void speechT5ArchitectureIsAudioModel() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"encoder_decoder","architectures":["SpeechT5Model"]}
                                """, ModelConfig.class);

                assertTrue(ModelRuntimeTraits.fallbackFromConfig(config).audioModel());
                assertFalse(ModelRuntimeTraits.fallbackFromConfig(config).visionModel());
        }

        @Test
        void conditionalGenerationArchitectureIsMultimodalButNotAudio() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"bart","architectures":["BartForConditionalGeneration"]}
                                """, ModelConfig.class);
                ModelRuntimeTraits traits = ModelRuntimeTraits.fallbackFromConfig(config);

                assertFalse(traits.audioModel());
                assertFalse(traits.visionModel());
                assertTrue(traits.multimodalModel());
        }

        @Test
        void fromConfigRemainsCompatibilityAliasForFallback() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}
                                """, ModelConfig.class);

                assertEquals(ModelRuntimeTraits.fallbackFromConfig(config), ModelRuntimeTraits.fromConfig(config));
        }

        @Test
        void manualRuntimeTraitsCanPreserveDetectedVisionModality() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"qwen2_vl","architectures":["Qwen2VLForConditionalGeneration"]}
                                """, ModelConfig.class);
                ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                                .qwenText()
                                .build()
                                .withDetectedModalities(config);

                assertFalse(traits.audioModel());
                assertTrue(traits.visionModel());
                assertTrue(traits.multimodalModel());
        }

        @Test
        void knownVlmFamilyNamesAreVisionModels() throws Exception {
                ModelConfig config = new ObjectMapper().readValue("""
                                {"model_type":"idefics3","architectures":["Idefics3ForConditionalGeneration"]}
                                """, ModelConfig.class);

                assertTrue(ModelRuntimeTraits.fallbackFromConfig(config).visionModel());
        }
}
