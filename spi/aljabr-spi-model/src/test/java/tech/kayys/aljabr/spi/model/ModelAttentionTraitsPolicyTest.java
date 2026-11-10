/*
 * Aljabr Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits.AttentionRuntimeTraits;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelAttentionTraitsPolicyTest {

  @Test
  void nativeBf16MatvecEnablesSpecialMetalAttentionPolicy() {
    AttentionRuntimeTraits traits = ModelAttentionTraitsPolicy.nativeBf16Matvec();

    assertTrue(traits.splitHalfRope());
    assertTrue(traits.attentionSoftCapAppliesToFinalLogitsOnly());
    assertTrue(traits.preferMetalPerHeadRmsNorm());
    assertTrue(traits.preferNativeMetalBf16Linear());
    assertTrue(traits.disallowBf16ToF16LinearConversion());
    assertTrue(traits.restrictLegacyMetalAttentionBridge());
    assertTrue(traits.supportsForcedDenseAttention());
  }

  @Test
  void qwenCompactConfigEnablesPagedMetalPrefillLimit() throws Exception {
    ModelConfig config = new ObjectMapper().readValue("""
        {
          "model_type": "qwen2",
          "num_hidden_layers": 24,
          "hidden_size": 1536,
          "intermediate_size": 8960
        }
        """, ModelConfig.class);
    AttentionRuntimeTraits traits = ModelAttentionTraitsPolicy.qwenText(config);

    assertTrue(traits.compactAttentionMatvecCandidate());
    assertEquals(
        ModelAttentionTraitsPolicy.DEFAULT_QWEN_PAGED_METAL_PREFILL_MAX_TOKENS,
        traits.defaultPagedMetalPrefillMaxTokens());
  }

  @Test
  void phiTextUsesPackedQkvProjection() throws Exception {
    ModelConfig config = new ObjectMapper().readValue("""
        {
          "model_type": "phi3",
          "num_hidden_layers": 32,
          "hidden_size": 3072,
          "intermediate_size": 8192
        }
        """, ModelConfig.class);
    AttentionRuntimeTraits traits = ModelAttentionTraitsPolicy.phiText(config);

    assertTrue(traits.packedQkvProjection());
    assertTrue(traits.largeAttentionMatvecCandidate());
  }

  @Test
  void genericLargeMatvecCandidateIsDisabledForPerLayerInputPath() throws Exception {
    ModelConfig config = new ObjectMapper().readValue("""
        {
          "model_type": "gemma",
          "num_hidden_layers": 34,
          "hidden_size": 3072,
          "intermediate_size": 8192
        }
        """, ModelConfig.class);

    assertTrue(ModelAttentionTraitsPolicy.generic(config, false).largeAttentionMatvecCandidate());
    assertFalse(ModelAttentionTraitsPolicy.generic(config, true).largeAttentionMatvecCandidate());
  }

  @Test
  void nestedRuntimeTraitFactoriesRemainCompatibilityFacades() throws Exception {
    ModelConfig config = new ObjectMapper().readValue("""
        {
          "model_type": "qwen2",
          "num_hidden_layers": 24,
          "hidden_size": 1536,
          "intermediate_size": 8960
        }
        """, ModelConfig.class);



  }
}
