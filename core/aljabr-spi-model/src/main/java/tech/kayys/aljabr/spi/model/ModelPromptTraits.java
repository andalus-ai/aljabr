/*
 * Aljabr Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.model.ModelRuntimeTraits.PromptBosPolicy;

import java.util.Locale;
import java.util.Set;

/**
 * Prompt and tokenizer-control policy derived from model family traits.
 *
 * <p>
 * This keeps prompt defaults, BOS insertion, and control-token validation
 * policy out of broader runtime traits so model-family prompt behavior can
 * evolve independently from attention and modality policy.
 * @author bhangun
 */
public record ModelPromptTraits(
        PromptBosPolicy promptBosPolicy,
        Set<String> allowedControlTokenTexts,
        boolean validateContinuationTokensByDecode,
        boolean rejectEmptyDecodedTokens,
        boolean skipDefaultSystemPromptInjection,
        String defaultSystemPrompt,
        Set<String> turnPromptPrefixes,
        boolean requiresChatTemplateFormatting) {

    public static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
    public static final String QWEN_DEFAULT_SYSTEM_PROMPT =
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    public static ModelPromptTraits fromConfig(ModelConfig config) {
        if (config == null) {
            return new ModelPromptTraits(PromptBosPolicy.DEFAULT, Set.of(), false, false, false, DEFAULT_SYSTEM_PROMPT, Set.of(), false);
        }
        String arch = (config.primaryArchitecture() != null ? config.primaryArchitecture() : "").toLowerCase(Locale.ROOT);
        String modelType = (config.modelType() != null ? config.modelType() : "").toLowerCase(Locale.ROOT);
        if (arch.contains("gemma4") || modelType.startsWith("gemma4")) {
            return new ModelPromptTraits(
                    PromptBosPolicy.NEVER,
                    Set.of("<|channel>", "<channel|>", "<|think|>", "<|turn>", "<turn|>"),
                    true,
                    true,
                    true,
                    DEFAULT_SYSTEM_PROMPT,
                    Set.of(),
                    false);
        }
        if (arch.contains("gemma") || modelType.contains("gemma")) {
            return new ModelPromptTraits(
                    PromptBosPolicy.TURN_AWARE,
                    Set.of(),
                    false,
                    false,
                    false,
                    DEFAULT_SYSTEM_PROMPT,
                    Set.of(),
                    false);
        }
        if (arch.contains("qwen") || modelType.contains("qwen")) {
            return new ModelPromptTraits(
                    PromptBosPolicy.DEFAULT,
                    Set.of("<|im_start|>", "<|im_end|>"),
                    false,
                    false,
                    false,
                    QWEN_DEFAULT_SYSTEM_PROMPT,
                    Set.of(),
                    true);
        }
        return new ModelPromptTraits(PromptBosPolicy.DEFAULT, Set.of(), false, false, false, DEFAULT_SYSTEM_PROMPT, Set.of(), false);
    }

    public ModelPromptTraits {
        promptBosPolicy = promptBosPolicy == null ? PromptBosPolicy.DEFAULT : promptBosPolicy;
        allowedControlTokenTexts = allowedControlTokenTexts == null
                ? Set.of()
                : Set.copyOf(allowedControlTokenTexts);
        defaultSystemPrompt = defaultSystemPrompt == null || defaultSystemPrompt.isBlank()
                ? DEFAULT_SYSTEM_PROMPT
                : defaultSystemPrompt;
        turnPromptPrefixes = turnPromptPrefixes == null
                ? Set.of()
                : Set.copyOf(turnPromptPrefixes);
    }
}
