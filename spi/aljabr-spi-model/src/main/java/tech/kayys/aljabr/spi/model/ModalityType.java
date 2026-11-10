package tech.kayys.aljabr.spi.model;

/**
 * Enumerates the supported input and output modality types for multimodal
 * requests.
 * @author bhangun
 */
public enum ModalityType {
    /** Plain text input or output. */
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    DOCUMENT,
    EMBEDDING,
    TIME_SERIES,
    THREE_DIMENSIONAL;

    /**
     * Returns true if this modality represents binary data (non-text).
     * 
     * @return true for IMAGE, AUDIO, VIDEO, DOCUMENT, EMBEDDING, TIME_SERIES, THREE_DIMENSIONAL
     */
    public boolean isBinary() {
        return this != TEXT;
    }
}
