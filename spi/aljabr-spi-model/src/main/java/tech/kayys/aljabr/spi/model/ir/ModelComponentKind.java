package tech.kayys.aljabr.spi.model.ir;

/** Standard semantic roles for model-level components. */
public enum ModelComponentKind {
    TOKENIZER,
    IMAGE_PROCESSOR,
    AUDIO_PROCESSOR,
    VIDEO_PROCESSOR,
    PREPROCESSOR,
    POSTPROCESSOR,
    MULTIMODAL_FUSION,
    CUSTOM
}
