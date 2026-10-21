/*
 * Aljabr Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits.PromptBosPolicy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelPromptTraitsTest {

    @Test
    void qwenUsesQwenDefaultSystemPrompt() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}
                """, ModelConfig.class);
        ModelPromptTraits traits = ModelPromptTraits.fromConfig(config);

        assertEquals(ModelPromptTraits.QWEN_DEFAULT_SYSTEM_PROMPT, traits.defaultSystemPrompt());
        assertEquals(PromptBosPolicy.DEFAULT, traits.promptBosPolicy());
        assertFalse(traits.skipDefaultSystemPromptInjection());
    }

    @Test
    void gemma4SkipsDefaultSystemPromptAndValidatesControlTokens() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"gemma4_text","architectures":["Gemma4ForCausalLM"]}
                """, ModelConfig.class);
        ModelPromptTraits traits = ModelPromptTraits.fromConfig(config);

        assertEquals(PromptBosPolicy.NEVER, traits.promptBosPolicy());
        assertTrue(traits.skipDefaultSystemPromptInjection());
        assertTrue(traits.validateContinuationTokensByDecode());
        assertTrue(traits.rejectEmptyDecodedTokens());
        assertTrue(traits.allowedControlTokenTexts().contains("<|channel>"));
    }

    @Test
    void gemmaFamilyUsesTurnAwareBosPolicy() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"gemma","architectures":["GemmaForCausalLM"]}
                """, ModelConfig.class);

        assertEquals(PromptBosPolicy.TURN_AWARE, ModelPromptTraits.fromConfig(config).promptBosPolicy());
    }
}
