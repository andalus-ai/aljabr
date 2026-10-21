package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

/**
 * Supported FFN activation functions.
 * @author bhangun
 */
public enum FFNActivationType {
    SILU, // LLaMA, Mistral
    GELU, // Gemma, Gemma-2
    RELU,
    GELU_QUICK
}
