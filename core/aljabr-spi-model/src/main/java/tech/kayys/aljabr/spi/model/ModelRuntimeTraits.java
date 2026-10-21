/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import java.util.Locale;
import java.util.Set;

/**
 * Runtime-only model family traits used by inference engines for policy
 * decisions that do not belong in tensor-name resolution.
 */
public record ModelRuntimeTraits(
        boolean nativeBf16Matvec,
        boolean geluGatedFfn,
        boolean perLayerInputEmbedding,
        boolean perLayerInputPath,
        PromptBosPolicy promptBosPolicy,
        Set<String> allowedControlTokenTexts,
        boolean validateContinuationTokensByDecode,
        boolean rejectEmptyDecodedTokens,
        Set<String> turnPromptPrefixes,
        AttentionRuntimeTraits attention,
        boolean audioModel,
        boolean visionModel,
        boolean multimodalModel,
        boolean qwenText) {/**
 * Enumeration of promptbospolicy values.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core enum operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */


    public enum PromptBosPolicy {
        DEFAULT,
        NEVER,
        TURN_AWARE
    }

    public record AttentionRuntimeTraits(
            boolean splitHalfRope,
            boolean attentionSoftCapAppliesToFinalLogitsOnly,
            boolean preferMetalPerHeadRmsNorm,
            boolean preferNativeMetalBf16Linear,
            boolean disallowBf16ToF16LinearConversion,
            boolean restrictLegacyMetalAttentionBridge,
            boolean supportsForcedDenseAttention,
            int defaultPagedMetalPrefillMaxTokens,
            boolean compactAttentionMatvecCandidate,
            boolean largeAttentionMatvecCandidate,
            boolean packedQkvProjection) {

        public static final AttentionRuntimeTraits EMPTY = ModelAttentionTraitsPolicy.empty();

        public static AttentionRuntimeTraits generic(ModelConfig config, boolean perLayerInputPath) {
            return ModelAttentionTraitsPolicy.generic(config, perLayerInputPath);
        }

        public static AttentionRuntimeTraits phiText(ModelConfig config) {
            return ModelAttentionTraitsPolicy.phiText(config);
        }

    }

    public static final String DEFAULT_SYSTEM_PROMPT = ModelPromptTraits.DEFAULT_SYSTEM_PROMPT;

    public static final ModelRuntimeTraits EMPTY = new ModelRuntimeTraits(false, false, false, false, PromptBosPolicy.DEFAULT, Set.of(), false, false, Set.of(), null, false, false, false, false);

    public static Builder builder() {
        return new Builder();
    }

    public static Builder builder(ModelRuntimeTraits traits) {
        return new Builder().copyFrom(traits);
    }

    public static ModelRuntimeTraits phiText(ModelConfig config) {
        return builder()
                .attention(AttentionRuntimeTraits.phiText(config))
                .build();
    }

    public ModelRuntimeTraits {
        promptBosPolicy = promptBosPolicy == null ? PromptBosPolicy.DEFAULT : promptBosPolicy;
        allowedControlTokenTexts = allowedControlTokenTexts == null
                ? Set.of()
                : Set.copyOf(allowedControlTokenTexts);
        attention = attention == null
                ? defaultAttentionTraits(nativeBf16Matvec, false, false, perLayerInputPath, null)
                : attention;
        multimodalModel = multimodalModel || audioModel || visionModel;
    }

    public ModelRuntimeTraits(boolean nativeBf16Matvec, boolean geluGatedFfn, boolean perLayerInputEmbedding,
            boolean perLayerInputPath) {
        this(nativeBf16Matvec, geluGatedFfn, perLayerInputEmbedding, perLayerInputPath,
                nativeBf16Matvec ? PromptBosPolicy.NEVER : (geluGatedFfn ? PromptBosPolicy.TURN_AWARE : PromptBosPolicy.DEFAULT),
                nativeBf16Matvec ? Set.of("<|channel>", "<channel|>", "<|think|>") : Set.of(),
                nativeBf16Matvec,
                nativeBf16Matvec,
                Set.of(),
                null,
                false,
                false,
                false,
                false);
    }

    public ModelRuntimeTraits(boolean nativeBf16Matvec, boolean geluGatedFfn, boolean perLayerInputEmbedding,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens) {
        this(nativeBf16Matvec, geluGatedFfn, perLayerInputEmbedding, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, Set.of(), null, false, false, false, false);
    }

    public ModelRuntimeTraits(boolean nativeBf16Matvec, boolean geluGatedFfn, boolean perLayerInputEmbedding,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens,
            AttentionRuntimeTraits attention) {
        this(nativeBf16Matvec, geluGatedFfn, perLayerInputEmbedding, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, Set.of(), attention, false, false, false, false);
    }

    public ModelRuntimeTraits(boolean nativeBf16Matvec, boolean geluGatedFfn, boolean perLayerInputEmbedding,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens,
            AttentionRuntimeTraits attention, boolean audioModel) {
        this(nativeBf16Matvec, geluGatedFfn, perLayerInputEmbedding, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, Set.of(), attention, audioModel, false, audioModel, false);
    }

    public ModelRuntimeTraits(boolean nativeBf16Matvec, boolean geluGatedFfn, boolean perLayerInputEmbedding,
            boolean perLayerInputPath, PromptBosPolicy promptBosPolicy, Set<String> allowedControlTokenTexts,
            boolean validateContinuationTokensByDecode, boolean rejectEmptyDecodedTokens,
            AttentionRuntimeTraits attention, boolean audioModel, boolean multimodalModel) {
        this(nativeBf16Matvec, geluGatedFfn, perLayerInputEmbedding, perLayerInputPath,
                promptBosPolicy, allowedControlTokenTexts,
                validateContinuationTokensByDecode, rejectEmptyDecodedTokens, Set.of(), attention,
                audioModel, false, multimodalModel, false);
    }

    /**
     * Compatibility alias for callers that still derive traits directly from
     * config metadata. New internal runtime paths should call
     * {@link #fallbackFromConfig(ModelConfig)} to make the fallback boundary
     * explicit.
     */
    public static ModelRuntimeTraits fromConfig(ModelConfig config) {
        return fallbackFromConfig(config);
    }

    /**
     * Derives coarse runtime traits directly from config metadata.
     *
     * <p>
     * This is intentionally a fallback path for generic loaders and legacy
     * adapters. Model-family modules should prefer overriding runtime traits so
     * tokenizer, attention, and modality policy stay owned by the farm.
     * </p>
     */
    public static ModelRuntimeTraits fallbackFromConfig(ModelConfig config) {
        if (config == null) {
            return EMPTY;
        }
        String modelType = normalizedModelType(config);
        boolean nativeBf16Matvec = modelType.startsWith("gemma4");
        boolean isGemma3Text = modelType.startsWith("gemma3");
        boolean isGemmaFamily = modelType.startsWith("gemma");
        boolean isQwenText = modelType.contains("qwen");
        PromptBosPolicy promptBosPolicy = nativeBf16Matvec ? PromptBosPolicy.NEVER : (isGemma3Text || isGemmaFamily ? PromptBosPolicy.TURN_AWARE : PromptBosPolicy.DEFAULT);
        ModelPromptTraits prompt = new ModelPromptTraits(
                promptBosPolicy,
                nativeBf16Matvec ? Set.of("<|channel>", "<channel|>", "<|think|>") : Set.of(),
                nativeBf16Matvec,
                nativeBf16Matvec,
                nativeBf16Matvec,
                isQwenText ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." : ModelPromptTraits.DEFAULT_SYSTEM_PROMPT,
                nativeBf16Matvec ? Set.of("<|turn>") : (isGemma3Text || isGemmaFamily ? Set.of("<start_of_turn>") : Set.of()),
                isGemma3Text || isQwenText);
        ModelModalityTraits modality = ModelModalityTraits.fromConfig(config);
        boolean perLayerInputPath = config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0;
        return builder()
                .nativeBf16Matvec(nativeBf16Matvec)
                .geluGatedFfn(isGemma3Text || nativeBf16Matvec)
                .perLayerInputEmbedding(nativeBf16Matvec && perLayerInputPath)
                .perLayerInputPath(perLayerInputPath)
                .prompt(prompt)
                .attention(defaultAttentionTraits(nativeBf16Matvec, isGemma3Text, isQwenText, perLayerInputPath, config))
                .modalities(modality)
                .build();
    }

    public ModelRuntimeTraits withDetectedModalities(ModelConfig config) {
        if (config == null) {
            return this;
        }
        ModelModalityTraits modality = ModelModalityTraits.fromConfig(config);
        if ((!modality.audioModel() || audioModel)
                && (!modality.visionModel() || visionModel)
                && (!modality.multimodalModel() || multimodalModel)) {
            return this;
        }
        return builder(this)
                .audioModel(audioModel || modality.audioModel())
                .visionModel(visionModel || modality.visionModel())
                .multimodalModel(multimodalModel || modality.multimodalModel())
                .build();
    }

    public static boolean detectAudioModel(ModelConfig config) {
        return ModelModalityTraits.detectAudioModel(config);
    }

    public static boolean detectVisionModel(ModelConfig config) {
        return ModelModalityTraits.detectVisionModel(config);
    }

    public static boolean detectMultimodalModel(ModelConfig config) {
        return ModelModalityTraits.detectMultimodalModel(config);
    }

    public static final String QWEN_DEFAULT_SYSTEM_PROMPT = ModelPromptTraits.QWEN_DEFAULT_SYSTEM_PROMPT;

    public boolean qwenText() {
        return qwenText;
    }

    public boolean gemma3Text() {
        return promptBosPolicy == PromptBosPolicy.TURN_AWARE
                && geluGatedFfn
                && attention != null
                && attention.splitHalfRope();
    }

    public boolean skipDefaultSystemPromptInjection() {
        return nativeBf16Matvec;
    }

    public String defaultSystemPrompt() {
        return qwenText() ? QWEN_DEFAULT_SYSTEM_PROMPT : ModelPromptTraits.DEFAULT_SYSTEM_PROMPT;
    }

    private static String normalizedModelType(ModelConfig config) {
        return config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
    }

    private static AttentionRuntimeTraits defaultAttentionTraits(boolean nativeBf16Matvec, boolean isGemma3Text,
            boolean isQwenText, boolean perLayerInputPath, ModelConfig config) {
        if (nativeBf16Matvec) {
            return new AttentionRuntimeTraits(true, true, true, true, true, true, true, 0, false, false, false);
        }
        if (isGemma3Text) {
            return new AttentionRuntimeTraits(true, false, false, false, false, false, false, 0, false, false, false);
        }
        if (isQwenText) {
            boolean compact = ModelAttentionTraitsPolicy.isCompactAttentionMatvecCandidate(config);
            return new AttentionRuntimeTraits(false, false, false, false, false, false, false, compact ? 128 : 0, compact, ModelAttentionTraitsPolicy.isLargeAttentionMatvecCandidate(config, false, false), false);
        }
        return ModelAttentionTraitsPolicy.generic(config, perLayerInputPath);
    }

    /**
     * Named builder for model-family profiles. This keeps farm-owned runtime
     * policy readable and avoids positional boolean mistakes as traits grow.
     */
    public static final class Builder {
        private boolean nativeBf16Matvec;
        private boolean geluGatedFfn;
        private boolean perLayerInputEmbedding;
        private boolean perLayerInputPath;
        private PromptBosPolicy promptBosPolicy;
        private Set<String> allowedControlTokenTexts;
        private boolean allowedControlTokenTextsSet;
        private boolean validateContinuationTokensByDecode;
        private boolean validateContinuationTokensByDecodeSet;
        private boolean rejectEmptyDecodedTokens;
        private boolean rejectEmptyDecodedTokensSet;
        private Set<String> turnPromptPrefixes;
        private boolean turnPromptPrefixesSet;
        private AttentionRuntimeTraits attention;
        private boolean audioModel;
        private boolean visionModel;
        private boolean multimodalModel;
        private boolean qwenText;

        private Builder() {
        }

        public Builder qwenText() {
            return qwenText(true);
        }

        public Builder qwenText(boolean qwenText) {
            this.qwenText = qwenText;
            return this;
        }

        public Builder nativeBf16Matvec() {
            return nativeBf16Matvec(true);
        }

        public Builder nativeBf16Matvec(boolean nativeBf16Matvec) {
            this.nativeBf16Matvec = nativeBf16Matvec;
            return this;
        }

        public Builder geluGatedFfn() {
            return geluGatedFfn(true);
        }

        public Builder geluGatedFfn(boolean geluGatedFfn) {
            this.geluGatedFfn = geluGatedFfn;
            return this;
        }

        public Builder perLayerInputEmbedding() {
            return perLayerInputEmbedding(true);
        }

        public Builder perLayerInputEmbedding(boolean perLayerInputEmbedding) {
            this.perLayerInputEmbedding = perLayerInputEmbedding;
            return this;
        }

        public Builder perLayerInputPath() {
            return perLayerInputPath(true);
        }

        public Builder perLayerInputPath(boolean perLayerInputPath) {
            this.perLayerInputPath = perLayerInputPath;
            return this;
        }

        public Builder prompt(ModelPromptTraits prompt) {
            if (prompt == null) {
                return this;
            }
            return promptBosPolicy(prompt.promptBosPolicy())
                    .allowedControlTokenTexts(prompt.allowedControlTokenTexts())
                    .validateContinuationTokensByDecode(prompt.validateContinuationTokensByDecode())
                    .rejectEmptyDecodedTokens(prompt.rejectEmptyDecodedTokens())
                    .turnPromptPrefixes(prompt.turnPromptPrefixes());
        }

        public Builder promptBosPolicy(PromptBosPolicy promptBosPolicy) {
            this.promptBosPolicy = promptBosPolicy;
            return this;
        }

        public Builder allowedControlTokenTexts(Set<String> allowedControlTokenTexts) {
            this.allowedControlTokenTexts = allowedControlTokenTexts == null
                    ? Set.of()
                    : Set.copyOf(allowedControlTokenTexts);
            this.allowedControlTokenTextsSet = true;
            return this;
        }

        public Builder validateContinuationTokensByDecode(boolean validateContinuationTokensByDecode) {
            this.validateContinuationTokensByDecode = validateContinuationTokensByDecode;
            this.validateContinuationTokensByDecodeSet = true;
            return this;
        }

        public Builder rejectEmptyDecodedTokens(boolean rejectEmptyDecodedTokens) {
            this.rejectEmptyDecodedTokens = rejectEmptyDecodedTokens;
            this.rejectEmptyDecodedTokensSet = true;
            return this;
        }

        public Builder turnPromptPrefixes(Set<String> turnPromptPrefixes) {
            this.turnPromptPrefixes = turnPromptPrefixes == null
                    ? Set.of()
                    : Set.copyOf(turnPromptPrefixes);
            this.turnPromptPrefixesSet = true;
            return this;
        }

        public Builder attention(AttentionRuntimeTraits attention) {
            this.attention = attention;
            return this;
        }

        public Builder audioModel() {
            return audioModel(true);
        }

        public Builder audioModel(boolean audioModel) {
            this.audioModel = audioModel;
            return this;
        }

        public Builder visionModel() {
            return visionModel(true);
        }

        public Builder visionModel(boolean visionModel) {
            this.visionModel = visionModel;
            return this;
        }

        public Builder multimodalModel() {
            return multimodalModel(true);
        }

        public Builder multimodalModel(boolean multimodalModel) {
            this.multimodalModel = multimodalModel;
            return this;
        }

        public Builder modalities(ModelModalityTraits modality) {
            if (modality == null) {
                return this;
            }
            return audioModel(modality.audioModel())
                    .visionModel(modality.visionModel())
                    .multimodalModel(modality.multimodalModel());
        }

        public ModelRuntimeTraits build() {
            return new ModelRuntimeTraits(
                    nativeBf16Matvec,
                    geluGatedFfn,
                    perLayerInputEmbedding,
                    perLayerInputPath,
                    promptBosPolicy == null
                            ? (nativeBf16Matvec ? PromptBosPolicy.NEVER : (geluGatedFfn ? PromptBosPolicy.TURN_AWARE : PromptBosPolicy.DEFAULT))
                            : promptBosPolicy,
                    allowedControlTokenTextsSet
                            ? allowedControlTokenTexts
                            : (nativeBf16Matvec ? Set.of("<|channel>", "<channel|>", "<|think|>") : Set.of()),
                    validateContinuationTokensByDecodeSet
                            ? validateContinuationTokensByDecode
                            : nativeBf16Matvec,
                    rejectEmptyDecodedTokensSet
                            ? rejectEmptyDecodedTokens
                            : nativeBf16Matvec,
                    turnPromptPrefixesSet
                            ? turnPromptPrefixes
                            : (nativeBf16Matvec ? Set.of("<|turn>") : Set.of()),
                    attention,
                    audioModel,
                    visionModel,
                    multimodalModel,
                    qwenText);
        }

        private Builder copyFrom(ModelRuntimeTraits traits) {
            if (traits == null) {
                return this;
            }
            return nativeBf16Matvec(traits.nativeBf16Matvec())
                    .geluGatedFfn(traits.geluGatedFfn())
                    .perLayerInputEmbedding(traits.perLayerInputEmbedding())
                    .perLayerInputPath(traits.perLayerInputPath())
                    .promptBosPolicy(traits.promptBosPolicy())
                    .allowedControlTokenTexts(traits.allowedControlTokenTexts())
                    .validateContinuationTokensByDecode(traits.validateContinuationTokensByDecode())
                    .rejectEmptyDecodedTokens(traits.rejectEmptyDecodedTokens())
                    .turnPromptPrefixes(traits.turnPromptPrefixes())
                    .attention(traits.attention())
                    .audioModel(traits.audioModel())
                    .visionModel(traits.visionModel())
                    .multimodalModel(traits.multimodalModel())
                    .qwenText(traits.qwenText());
        }
    }
}
