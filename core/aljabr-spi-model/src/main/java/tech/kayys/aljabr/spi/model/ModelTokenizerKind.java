package tech.kayys.aljabr.spi.model;

/**
 * Tokenizer families advertised by detachable model-family plugins.
 *
 * <p>
 * The enum deliberately lives in the model SPI rather than tokenizer-core so
 * model-family JARs can advertise tokenizer compatibility without depending on
 * a concrete tokenizer runtime.
 * </p>
 * @author bhangun
 */
public enum ModelTokenizerKind {
    HUGGING_FACE_BPE,
    SENTENCE_PIECE_BPE,
    WORD_PIECE,
    TIKTOKEN,
    CUSTOM
}
