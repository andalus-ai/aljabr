package tech.kayys.aljabr.cache;

import java.util.HashSet;
import java.util.Objects;

/**
 * Immutable selection of absolute sequence positions for sparse attention.
 *
 * <p>The selection is independent of a particular model, indexer, attention
 * kernel, cache format, or speculative-decoding algorithm. Position order is
 * preserved because an indexer or consumer may attach meaning to that order.
 * Duplicate and out-of-domain positions are rejected.</p>
 */
public final class AttentionIndexSelection {

    private final int sourceSequenceLength;
    private final int[] positions;

    /**
     * Creates a selection over the prefix {@code [0, sourceSequenceLength)}.
     *
     * @param sourceSequenceLength length of the sequence searched by the indexer
     * @param positions selected absolute positions, in consumer-defined order
     */
    public AttentionIndexSelection(int sourceSequenceLength, int[] positions) {
        if (sourceSequenceLength < 0) {
            throw new IllegalArgumentException("sourceSequenceLength must not be negative");
        }
        Objects.requireNonNull(positions, "positions");
        this.sourceSequenceLength = sourceSequenceLength;
        this.positions = positions.clone();

        HashSet<Integer> seen = new HashSet<>();
        for (int position : this.positions) {
            if (position < 0 || position >= sourceSequenceLength) {
                throw new IllegalArgumentException(
                        "Selected position " + position
                                + " is outside source sequence length " + sourceSequenceLength);
            }
            if (!seen.add(position)) {
                throw new IllegalArgumentException(
                        "Duplicate attention position: " + position);
            }
        }
    }

    /** Length of the sequence from which these positions were selected. */
    public int sourceSequenceLength() {
        return sourceSequenceLength;
    }

    /** Number of selected positions. */
    public int size() {
        return positions.length;
    }

    /** Returns the selected absolute position at the given selection offset. */
    public int positionAt(int selectionOffset) {
        return positions[selectionOffset];
    }

    /** Returns a defensive copy in the original selection order. */
    public int[] positions() {
        return positions.clone();
    }

    /**
     * Verifies that this selection can be applied to the supplied cache view.
     * A longer view is valid; a shorter view cannot satisfy the selection's
     * source-sequence contract.
     */
    public void validateAgainst(KVCacheReadView view) {
        Objects.requireNonNull(view, "view");
        if (sourceSequenceLength > view.visibleSequenceLength()) {
            throw new IllegalArgumentException(
                    "Selection requires " + sourceSequenceLength
                            + " visible positions but the KV view exposes "
                            + view.visibleSequenceLength());
        }
    }
}
