/*
 * Aljabr — SafeTensors quantization
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Objects;

/**
 * Explicit mapping between the neutral strategy vocabulary and the
 * historical engine-scoped enum retained for source compatibility.
 *
 * <p>Keep this adapter at the compatibility boundary. The neutral
 * {@link QuantizationStrategy} deliberately has no dependency on the
 * orchestration engine.</p>
 */
public final class QuantizationStrategyCompatibility {

    private QuantizationStrategyCompatibility() {
    }

    public static QuantizationStrategy fromLegacy(QuantizationEngine.QuantStrategy strategy) {
        return QuantizationStrategy.valueOf(Objects.requireNonNull(strategy, "strategy").name());
    }

    public static QuantizationEngine.QuantStrategy toLegacy(QuantizationStrategy strategy) {
        return QuantizationEngine.QuantStrategy.valueOf(
                Objects.requireNonNull(strategy, "strategy").name());
    }
}
