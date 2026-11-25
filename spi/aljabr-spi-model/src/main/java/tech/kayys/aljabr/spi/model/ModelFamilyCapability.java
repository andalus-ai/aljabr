package tech.kayys.aljabr.spi.model;

/**
 * Capabilities that a model-family plugin may declare.
 * These capability flags drive contract validation and runtime routing.
 */
public enum ModelFamilyCapability {
    /** The family supports causal-language-model (text-generation) inference. */
    CAUSAL_LM,
    TOKENIZER,
    MULTIMODAL,
    DIRECT_SAFETENSOR_INFERENCE,
    EMBEDDING,
    VISION,
    CHAT_TEMPLATE,
    AUDIO,
    MOE,
    /** The family performs forecasting over ordered temporal observations. */
    TIME_SERIES_FORECASTING
}
