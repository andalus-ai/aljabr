package tech.kayys.aljabr.gguf.api;

/**
 * A data modality accepted or produced by a GGUF model/provider.
 *
 * <p>These identifiers describe capability dimensions only. Declaring a
 * modality does not imply that a particular model or provider supports it;
 * support must be established by its capability profile.</p>
 */
public enum GgufModality {
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    STRUCTURED
}
