/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;

/**
 * Pure policy for selecting the effective strategy and eligible tensors for
 * inference-time weight quantization.
 *
 * <p>This class deliberately owns no registry, cache, executor, logging, or
 * framework lifecycle. The runtime engine remains responsible for those
 * operational concerns.</p>
 */
public final class InferenceQuantizationPolicy {

    private InferenceQuantizationPolicy() {
    }

    /**
     * Resolves strategies that currently have an inference-compatible path.
     * A null strategy means quantization is disabled.
     */
    public static QuantizationStrategy normalizeStrategy(QuantizationStrategy strategy) {
        if (strategy == null) {
            return QuantizationStrategy.NONE;
        }
        return switch (strategy) {
            case TURBO -> QuantizationStrategy.BNB;
            case GPTQ -> QuantizationStrategy.INT4;
            default -> strategy;
        };
    }

    /**
     * Determines whether a named model tensor is eligible for inference
     * quantization. Embeddings and vision tower weights intentionally pass
     * through unchanged.
     */
    public static boolean shouldQuantize(String name, AccelTensor tensor) {
        return tensor != null
                && name != null
                && (name.endsWith(".weight") || name.endsWith(".qweight")
                        || name.endsWith("down_proj") || name.endsWith("gate_up_proj"))
                && (tensor.rank() == 2 || tensor.rank() == 3)
                && tensor.numel() >= 4096
                && !name.contains("embed_tokens")
                && !name.contains("position_embedding")
                && !name.contains("vision_tower");
    }

    /**
     * Creates the built-in default configuration for a normalized strategy.
     */
    public static QuantConfig defaultConfig(QuantizationStrategy strategy) {
        Objects.requireNonNull(strategy, "strategy cannot be null");
        return switch (strategy) {
            case INT4 -> QuantConfig.int4Gptq();
            case INT8 -> QuantConfig.int8();
            case FP8 -> QuantConfig.fp8();
            case TURBO -> QuantConfig.turbo();
            case AWQ -> QuantConfig.awq(4, 128);
            case GPTQ -> QuantConfig.builder().quantizationStrategy(QuantizationStrategy.GPTQ).build();
            case BNB -> QuantConfig.bnb();
            case NONE -> new QuantConfig();
        };
    }
}
