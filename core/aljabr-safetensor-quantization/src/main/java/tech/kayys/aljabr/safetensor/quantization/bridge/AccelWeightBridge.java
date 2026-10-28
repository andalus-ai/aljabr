/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compatibility adapter. The canonical safetensor-to-AccelTensor bridge lives
 * in aljabr-safetensor-loader so default inference does not need the optional
 * Tafkir-backed quantization module.
 */
package tech.kayys.aljabr.safetensor.quantization.bridge;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;
import java.util.Map;

@ApplicationScoped
@Deprecated(forRemoval = false)
public class AccelWeightBridge {
    private final tech.kayys.aljabr.safetensor.loader.bridge.AccelWeightBridge delegate =
            new tech.kayys.aljabr.safetensor.loader.bridge.AccelWeightBridge();

    public AccelTensor bridge(SafetensorTensor tensor) {
        return delegate.bridge(tensor);
    }

    public Map<String, AccelTensor> bridgeAll(SafetensorShardSession session) {
        return delegate.bridgeAll(session);
    }
}
