/*
 * Aljabr — SafeTensors quantization
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;

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
    BNB;

    /**
     * Maps the legacy engine-scoped strategy to the neutral strategy.
     *
     * @param strategy legacy strategy
     * @return equivalent neutral strategy
     */
    public static QuantizationStrategy fromLegacy(QuantizationEngine.QuantStrategy strategy) {
        return valueOf(Objects.requireNonNull(strategy, "strategy").name());
    }

    /**
     * Maps this strategy to the legacy engine-scoped API.
     *
     * @return equivalent legacy strategy
     */
    public QuantizationEngine.QuantStrategy toLegacy() {
        return QuantizationEngine.QuantStrategy.valueOf(name());
    }
}
