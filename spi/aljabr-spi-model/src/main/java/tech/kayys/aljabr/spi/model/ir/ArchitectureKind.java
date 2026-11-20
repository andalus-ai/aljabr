package tech.kayys.aljabr.spi.model.ir;

/**
 * Broad architectural category. This classification is descriptive, not an
 * execution constraint; concrete architecture identifiers remain extensible.
 */
public enum ArchitectureKind {
    DECODER_ONLY_TRANSFORMER,
    ENCODER_ONLY_TRANSFORMER,
    ENCODER_DECODER_TRANSFORMER,
    VISION,
    CONVOLUTIONAL,
    DIFFUSION,
    SPEECH,
    MULTIMODAL,
    EMBEDDING,
    RERANKER,
    RECOMMENDER,
    CLASSICAL,
    CUSTOM
}
