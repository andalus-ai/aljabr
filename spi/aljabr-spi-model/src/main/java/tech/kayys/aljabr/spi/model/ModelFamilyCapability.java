package tech.kayys.aljabr.spi.model;

/**
 * Enumeration of capabilities that a {@link ModelFamilyPlugin} may declare.
 * These capability flags drive contract validation and runtime routing.
 * @author bhangun
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
    MOE
}
