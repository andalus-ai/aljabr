/*
 * Aljabr Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelModalityTraitsTest {

    @Test
    void whisperConfigIsAudioButNotVision() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}
                """, ModelConfig.class);
        ModelModalityTraits traits = ModelModalityTraits.fromConfig(config);

        assertTrue(traits.audioModel());
        assertFalse(traits.visionModel());
        assertTrue(traits.multimodalModel());
    }

    @Test
    void qwenVlConfigIsVisionButNotAudio() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"qwen2_vl","architectures":["Qwen2VLForConditionalGeneration"]}
                """, ModelConfig.class);
        ModelModalityTraits traits = ModelModalityTraits.fromConfig(config);

        assertFalse(traits.audioModel());
        assertTrue(traits.visionModel());
        assertTrue(traits.multimodalModel());
    }

    @Test
    void conditionalGenerationIsMultimodalWithoutSpecificModality() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"bart","architectures":["BartForConditionalGeneration"]}
                """, ModelConfig.class);
        ModelModalityTraits traits = ModelModalityTraits.fromConfig(config);

        assertFalse(traits.audioModel());
        assertFalse(traits.visionModel());
        assertTrue(traits.multimodalModel());
    }

    @Test
    void gemma4UnifiedConfigIsAudioVisionAndMultimodal() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"]}
                """, ModelConfig.class);
        ModelModalityTraits traits = ModelModalityTraits.fromConfig(config);

        assertTrue(traits.audioModel());
        assertTrue(traits.visionModel());
        assertTrue(traits.multimodalModel());
    }
}
