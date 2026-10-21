/*
 * Aljabr Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import java.util.Locale;

/**
 * Runtime-only modality traits derived from model configuration.
 *
 * <p>
 * This keeps audio, vision, and broad multimodal classification separate from
 * prompt, attention, and tensor-runtime policy. Model-family modules can evolve
 * modality rules here without growing {@link ModelRuntimeTraits}.
 * @author bhangun
 */
public record ModelModalityTraits(
        boolean audioModel,
        boolean visionModel,
        boolean multimodalModel) {

    public static final ModelModalityTraits EMPTY = new ModelModalityTraits(false, false, false);

    public ModelModalityTraits {
        multimodalModel = multimodalModel || audioModel || visionModel;
    }

    public static ModelModalityTraits fromConfig(ModelConfig config) {
        if (config == null) {
            return EMPTY;
        }
        String modelType = normalizedModelType(config);
        String architecture = normalizedArchitecture(config);
        boolean audioModel = isAudioModel(modelType, architecture);
        boolean visionModel = isVisionModel(modelType, architecture);
        return new ModelModalityTraits(
                audioModel,
                visionModel,
                isMultimodalModel(modelType, architecture, audioModel, visionModel));
    }

    public static boolean detectAudioModel(ModelConfig config) {
        return fromConfig(config).audioModel();
    }

    public static boolean detectVisionModel(ModelConfig config) {
        return fromConfig(config).visionModel();
    }

    public static boolean detectMultimodalModel(ModelConfig config) {
        return fromConfig(config).multimodalModel();
    }

    private static boolean isAudioModel(String modelType, String architecture) {
        return isGemma4Unified(modelType, architecture)
                || modelType.contains("speecht5")
                || modelType.contains("speech_t5")
                || modelType.contains("whisper")
                || architecture.contains("speecht5")
                || architecture.contains("speech_t5")
                || architecture.contains("whisper");
    }

    private static boolean isVisionModel(String modelType, String architecture) {
        return isGemma4Unified(modelType, architecture)
                || modelType.contains("vision")
                || modelType.contains("video")
                || modelType.contains("image")
                || modelType.contains("multimodal")
                || modelType.contains("vl")
                || modelType.contains("ocr")
                || modelType.contains("llava")
                || modelType.contains("idefics")
                || modelType.contains("paligemma")
                || modelType.contains("pixtral")
                || modelType.contains("mllama")
                || modelType.contains("florence")
                || modelType.contains("blip")
                || modelType.contains("chameleon")
                || modelType.contains("kosmos")
                || architecture.contains("vision")
                || architecture.contains("video")
                || architecture.contains("image")
                || architecture.contains("multimodal")
                || architecture.contains("vl")
                || architecture.contains("ocr")
                || architecture.contains("llava")
                || architecture.contains("idefics")
                || architecture.contains("paligemma")
                || architecture.contains("pixtral")
                || architecture.contains("mllama")
                || architecture.contains("florence")
                || architecture.contains("blip")
                || architecture.contains("chameleon")
                || architecture.contains("kosmos");
    }

    private static boolean isGemma4Unified(String modelType, String architecture) {
        return "gemma4_unified".equals(modelType)
                || architecture.contains("gemma4unified")
                || architecture.contains("gemma4formultimodallm");
    }

    private static boolean isMultimodalModel(String modelType, String architecture, boolean audioModel,
            boolean visionModel) {
        return audioModel
                || visionModel
                || modelType.contains("audio")
                || architecture.contains("conditionalgeneration")
                || architecture.contains("audio");
    }

    private static String normalizedModelType(ModelConfig config) {
        return config.modelType() == null ? "" : config.modelType().toLowerCase(Locale.ROOT);
    }

    private static String normalizedArchitecture(ModelConfig config) {
        return config.primaryArchitecture() == null
                ? ""
                : config.primaryArchitecture().toLowerCase(Locale.ROOT);
    }
}
