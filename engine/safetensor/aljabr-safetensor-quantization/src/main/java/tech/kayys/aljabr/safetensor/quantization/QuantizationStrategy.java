/*
 * Aljabr — SafeTensors quantization
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

/**
 * Format- and provider-neutral quantization strategy identifier.
 *
 * <p>This type is the stable strategy vocabulary for quantization
 * configuration. Runtime orchestration may adapt it to its own lifecycle
 * and execution model.</p>
 */
public enum QuantizationStrategy {
    NONE,
    INT4,
    INT8,
    FP8,
    TURBO,
    AWQ,
    GPTQ,
    BNB
}
