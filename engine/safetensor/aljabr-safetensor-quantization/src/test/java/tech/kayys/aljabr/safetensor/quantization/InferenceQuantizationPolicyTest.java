/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class InferenceQuantizationPolicyTest {

    @Test
    void nullStrategyDisablesInferenceQuantization() {
        assertEquals(QuantizationStrategy.NONE, InferenceQuantizationPolicy.normalizeStrategy(null));
    }

    @Test
    void unsupportedDirectStrategiesResolveToCompatiblePaths() {
        assertEquals(QuantizationStrategy.BNB,
                InferenceQuantizationPolicy.normalizeStrategy(QuantizationStrategy.TURBO));
        assertEquals(QuantizationStrategy.INT4,
                InferenceQuantizationPolicy.normalizeStrategy(QuantizationStrategy.GPTQ));
    }

    @Test
    void supportedStrategiesRemainUnchanged() {
        assertEquals(QuantizationStrategy.INT8,
                InferenceQuantizationPolicy.normalizeStrategy(QuantizationStrategy.INT8));
        assertEquals(QuantizationStrategy.FP8,
                InferenceQuantizationPolicy.normalizeStrategy(QuantizationStrategy.FP8));
    }

    @Test
    void defaultConfigMatchesNormalizedStrategy() {
        assertEquals(QuantizationStrategy.INT4,
                InferenceQuantizationPolicy.defaultConfig(QuantizationStrategy.INT4).getQuantizationStrategy());
        assertEquals(QuantizationStrategy.BNB,
                InferenceQuantizationPolicy.defaultConfig(QuantizationStrategy.BNB).getQuantizationStrategy());
        assertThrows(NullPointerException.class,
                () -> InferenceQuantizationPolicy.defaultConfig(null));
    }

    @Test
    void nullTensorAndNullNameAreNeverEligible() {
        assertFalse(InferenceQuantizationPolicy.shouldQuantize("layer.weight", null));
        assertFalse(InferenceQuantizationPolicy.shouldQuantize(null, null));
    }
}
