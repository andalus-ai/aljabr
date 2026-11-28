package tech.kayys.aljabr.cache;

/**
 * Strategy for selecting sequence positions that an attention operation may
 * inspect.
 *
 * <p>An implementation may use learned sparse indexers, heuristics, routing
 * metadata, locality, or backend-specific algorithms. It returns positions
 * only; it does not read KV data or compute attention. Implementations must
 * return an immutable {@link AttentionIndexSelection} compatible with the
 * supplied request.</p>
 */
@FunctionalInterface
public interface AttentionIndexer {

    /**
     * Selects positions for one attention query.
     *
     * @param request validated query and source-prefix constraints
     * @return selected absolute sequence positions
     */
    AttentionIndexSelection select(AttentionIndexingRequest request);
}
