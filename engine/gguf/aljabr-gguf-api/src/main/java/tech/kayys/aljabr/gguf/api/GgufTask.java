package tech.kayys.aljabr.gguf.api;

/**
 * A task that a GGUF model/provider may implement.
 *
 * <p>Task identifiers are intentionally independent of model families and
 * execution runtimes. Additions should be backward-compatible.</p>
 */
public enum GgufTask {
    TEXT_GENERATION,
    IMAGE_UNDERSTANDING,
    AUDIO_PERCEPTION,
    AUDIO_TRANSCRIPTION,
    SPEECH_GENERATION,
    AUDIO_TO_AUDIO,
    VIDEO_UNDERSTANDING,
    EMBEDDING
}
