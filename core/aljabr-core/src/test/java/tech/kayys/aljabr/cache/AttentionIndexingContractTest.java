package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AttentionIndexingContractTest {

    @Test
    void requestAcceptsValidNonSortedSelection() {
        AttentionIndexingRequest request =
                new AttentionIndexingRequest(8, 5, 4, true);
        AttentionIndexSelection selection =
                new AttentionIndexSelection(8, new int[] {5, 1, 3, 0});

        assertDoesNotThrow(() -> request.validate(selection));
    }

    @Test
    void requestRejectsSelectionFromDifferentSourceLength() {
        AttentionIndexingRequest request =
                new AttentionIndexingRequest(8, 5, 4, true);
        AttentionIndexSelection selection =
                new AttentionIndexSelection(7, new int[] {5, 1});

        assertThrows(IllegalArgumentException.class,
                () -> request.validate(selection));
    }

    @Test
    void requestRejectsSelectionOverBudget() {
        AttentionIndexingRequest request =
                new AttentionIndexingRequest(8, 5, 2, true);
        AttentionIndexSelection selection =
                new AttentionIndexSelection(8, new int[] {5, 1, 3});

        assertThrows(IllegalArgumentException.class,
                () -> request.validate(selection));
    }

    @Test
    void causalRequestRejectsFuturePositions() {
        AttentionIndexingRequest request =
                new AttentionIndexingRequest(8, 3, 4, true);
        AttentionIndexSelection selection =
                new AttentionIndexSelection(8, new int[] {3, 1, 4});

        assertThrows(IllegalArgumentException.class,
                () -> request.validate(selection));
    }

    @Test
    void nonCausalRequestAllowsFuturePositions() {
        AttentionIndexingRequest request =
                new AttentionIndexingRequest(8, 3, 4, false);
        AttentionIndexSelection selection =
                new AttentionIndexSelection(8, new int[] {6, 1, 4});

        assertDoesNotThrow(() -> request.validate(selection));
    }

    @Test
    void requestRejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionIndexingRequest(0, 0, 1, true));
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionIndexingRequest(4, 4, 1, true));
        assertThrows(IllegalArgumentException.class,
                () -> new AttentionIndexingRequest(4, 0, 0, true));
    }

    @Test
    void indexerContractCanBeImplementedAsReusableStrategy() {
        AttentionIndexer indexer = request -> new AttentionIndexSelection(
                request.sourceSequenceLength(),
                new int[] {request.queryPosition(), 0});

        AttentionIndexingRequest request =
                new AttentionIndexingRequest(8, 5, 2, true);
        AttentionIndexSelection selection = indexer.select(request);

        assertDoesNotThrow(() -> request.validate(selection));
        assertArrayEquals(new int[] {5, 0}, selection.positions());
    }
}
