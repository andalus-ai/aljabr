package tech.kayys.aljabr.models.embeddinggemma2.input;

/** Role of one text input in a task-steered embedding request. */
public enum EmbeddingGemma2InputRole {
    /** Query/claim side of an asymmetric task. */
    QUERY,
    /** Document/evidence/code side of an asymmetric task. */
    DOCUMENT,
    /** Symmetric task input; the same instruction is used for both sides. */
    SINGLE
}
