package tech.kayys.aljabr.diffusion.video;

import java.util.Iterator;
import java.util.Objects;

/**
 * Consumes resource-owning video batches one at a time.
 *
 * <p>Each batch is processed inside a try-with-resources scope and closed
 * before the iterator is asked for the next batch. The consumer therefore
 * holds at most one batch returned by the source at a time. It does not
 * retain completed batches or own the source sequence.</p>
 */
public final class VideoFrameBatchConsumer {
    private VideoFrameBatchConsumer() {
    }

    /**
     * Processes and closes each batch before requesting the next one.
     *
     * <p>Uses default controls: no cancellation and no progress listener.</p>
     */
    public static <B extends AutoCloseable> long consume(
            Iterable<? extends B> batches,
            BatchProcessor<? super B> processor) throws Exception {
        return consume(batches, processor, VideoBatchProcessingContext.defaults());
    }

    /**
     * Processes resource-owning batches with synchronous cancellation and
     * progress reporting.
     *
     * <p>Cancellation is checked before requesting each next batch, before
     * processing it, and after the processor returns. A callback already in
     * progress is not interrupted. If cancellation is observed while a batch
     * is open, that batch is still closed and is not counted as completed.
     * Progress is reported only after successful processing and close. A
     * progress-listener failure stops consumption after the batch is closed.</p>
     *
     * <p>If processing fails, the current batch is still closed and any
     * close failure is attached to the processing failure as a suppressed
     * exception. If closing fails after successful processing, iteration
     * stops and the close failure is propagated.</p>
     *
     * @param batches lazy source of resource-owning batches
     * @param processor operation to perform for each batch
     * @param context cancellation and progress controls for this run
     * @param <B> batch type
     * @return number of batches successfully processed and closed
     * @throws Exception if processing or closing a batch fails
     */
    public static <B extends AutoCloseable> long consume(
            Iterable<? extends B> batches,
            BatchProcessor<? super B> processor,
            VideoBatchProcessingContext context) throws Exception {
        Objects.requireNonNull(batches, "batches");
        Objects.requireNonNull(processor, "processor");
        Objects.requireNonNull(context, "context");

        long processed = 0;
        Iterator<? extends B> iterator = batches.iterator();
        while (true) {
            context.throwIfCancellationRequested();
            if (!iterator.hasNext()) {
                return processed;
            }
            context.throwIfCancellationRequested();

            B batch = Objects.requireNonNull(iterator.next(), "batch");
            try (B ownedBatch = batch) {
                context.throwIfCancellationRequested();
                processor.process(ownedBatch);
                context.throwIfCancellationRequested();
            }
            processed = Math.incrementExact(processed);
            context.reportProgress(processed);
        }
    }

    /** A batch operation that may fail with a checked exception. */
    @FunctionalInterface
    public interface BatchProcessor<B> {
        void process(B batch) throws Exception;
    }
}
