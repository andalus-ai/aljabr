package tech.kayys.aljabr.cache;

/**
 * Optional model/runtime capability for exposing sparse-attention indexing.
 *
 * <p>Models without a compatible sparse indexer do not implement this
 * capability. Callers must retain dense-attention or another supported
 * fallback when it is absent.</p>
 */
public interface AttentionIndexingCapability {

    /** Returns the indexer associated with this capability. */
    AttentionIndexer attentionIndexer();
}
