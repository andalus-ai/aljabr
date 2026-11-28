package tech.kayys.aljabr.diffusion.video;

import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * Immutable, synchronous controls for one video batch-consumption run.
 *
 * <p>The cancellation signal is polled between batches and immediately after
 * each processor callback. It does not interrupt a callback already in
 * progress. Progress is reported only after a batch has been processed and
 * successfully closed. Unknown source lengths are represented explicitly.</p>
 */
public final class VideoBatchProcessingContext {
    private static final BooleanSupplier NEVER_CANCELLED = () -> false;
    private static final Consumer<VideoBatchProgress> NO_PROGRESS = ignored -> { };

    private final BooleanSupplier cancellationRequested;
    private final OptionalLong totalBatches;
    private final Consumer<VideoBatchProgress> progressListener;

    private VideoBatchProcessingContext(
            BooleanSupplier cancellationRequested,
            OptionalLong totalBatches,
            Consumer<VideoBatchProgress> progressListener) {
        this.cancellationRequested = Objects.requireNonNull(cancellationRequested, "cancellationRequested");
        this.totalBatches = Objects.requireNonNull(totalBatches, "totalBatches");
        this.progressListener = Objects.requireNonNull(progressListener, "progressListener");
        if (totalBatches.isPresent() && totalBatches.getAsLong() < 0) {
            throw new IllegalArgumentException("totalBatches must not be negative");
        }
    }

    /** Context with no cancellation and no progress callback. */
    public static VideoBatchProcessingContext defaults() {
        return new VideoBatchProcessingContext(NEVER_CANCELLED, OptionalLong.empty(), NO_PROGRESS);
    }

    /**
     * Creates run controls. A null-free optional total may be empty when the
     * source is lazy, streaming, or otherwise has no known finite length.
     */
    public static VideoBatchProcessingContext create(
            BooleanSupplier cancellationRequested,
            OptionalLong totalBatches,
            Consumer<VideoBatchProgress> progressListener) {
        return new VideoBatchProcessingContext(cancellationRequested, totalBatches, progressListener);
    }

    /** Throws when cancellation has been requested. */
    public void throwIfCancellationRequested() {
        if (cancellationRequested.getAsBoolean()) {
            throw new CancellationException("Video batch processing was cancelled");
        }
    }

    OptionalLong totalBatches() {
        return totalBatches;
    }

    void reportProgress(long completedBatches) {
        progressListener.accept(new VideoBatchProgress(completedBatches, totalBatches));
    }
}
