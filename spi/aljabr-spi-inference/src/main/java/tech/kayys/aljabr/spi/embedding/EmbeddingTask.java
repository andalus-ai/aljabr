package tech.kayys.aljabr.spi.embedding;

/**
 * Semantic intent for an embedding request.
 *
 * <p>Tasks describe how the model should interpret an input, not how the
 * inference runtime should schedule or transport it. Model-family runners
 * may reject tasks they do not support.</p>
 */
public enum EmbeddingTask {
    /** Preserve model-specific or legacy default behavior. */
    UNSPECIFIED,
    /** Encode a query for asymmetric information retrieval. */
    RETRIEVAL_QUERY,
    /** Encode a document or passage for asymmetric information retrieval. */
    RETRIEVAL_DOCUMENT,
    /** Encode inputs for code retrieval. */
    CODE_RETRIEVAL,
    /** Encode inputs for fact checking. */
    FACT_CHECKING,
    /** Encode inputs for pairwise or sentence-level semantic similarity. */
    SEMANTIC_SIMILARITY,
    /** Encode inputs for downstream classification. */
    CLASSIFICATION,
    /** Encode inputs for clustering. */
    CLUSTERING
}
