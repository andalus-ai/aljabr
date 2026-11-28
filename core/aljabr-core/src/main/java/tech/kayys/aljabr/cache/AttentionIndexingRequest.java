package tech.kayys.aljabr.cache;

/**
 * Model-neutral description of one sparse-attention index selection request.
 *
 * <p>The request describes the visible source prefix and query position. It
 * deliberately does not prescribe a scoring function, index data structure,
 * attention head layout, or model family. Callers that require distinct
 * per-layer or per-head selections issue distinct requests through the
 * appropriate model capability.</p>
 */
public record AttentionIndexingRequest(
        int sourceSequenceLength,
        int queryPosition,
        int maxSelectedPositions,
        boolean causal) {

    public AttentionIndexingRequest {
        if (sourceSequenceLength <= 0) {
            throw new IllegalArgumentException("sourceSequenceLength must be positive");
        }
        if (queryPosition < 0 || queryPosition >= sourceSequenceLength) {
            throw new IllegalArgumentException(
                    "queryPosition must be within the source sequence");
        }
        if (maxSelectedPositions <= 0) {
            throw new IllegalArgumentException("maxSelectedPositions must be positive");
        }
    }

    /**
     * Validates an indexer's result against this request.
     *
     * <p>Selection order is preserved and is not required to be sorted. A
     * causal request may select only positions at or before the query.</p>
     */
    public void validate(AttentionIndexSelection selection) {
        if (selection == null) {
            throw new IllegalArgumentException("selection must not be null");
        }
        if (selection.sourceSequenceLength() != sourceSequenceLength) {
            throw new IllegalArgumentException(
                    "Selection source length does not match indexing request");
        }
        if (selection.size() > maxSelectedPositions) {
            throw new IllegalArgumentException(
                    "Indexer returned " + selection.size()
                            + " positions; maximum is " + maxSelectedPositions);
        }
        if (causal) {
            for (int i = 0; i < selection.size(); i++) {
                if (selection.positionAt(i) > queryPosition) {
                    throw new IllegalArgumentException(
                            "Causal selection contains a future position: "
                                    + selection.positionAt(i));
                }
            }
        }
    }
}
