package tech.kayys.aljabr.diffusion.video;

import java.util.Objects;
import java.util.OptionalLong;

/**
 * Snapshot of successfully processed and released batches.
 *
 * @param completedBatches number of batches processed and closed
 * @param totalBatches known total, or empty when the source length is unknown
 */
public record VideoBatchProgress(long completedBatches, OptionalLong totalBatches) {
    public VideoBatchProgress {
        if (completedBatches < 0) {
            throw new IllegalArgumentException("completedBatches must not be negative");
        }
        totalBatches = Objects.requireNonNull(totalBatches, "totalBatches");
        if (totalBatches.isPresent() && totalBatches.getAsLong() < completedBatches) {
            throw new IllegalArgumentException("totalBatches must not be less than completedBatches");
        }
    }

    public boolean totalKnown() {
        return totalBatches.isPresent();
    }
}
