package tech.kayys.aljabr.diffusion.video;

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
     * <p>If processing fails, the current batch is still closed and any
     * close failure is attached to the processing failure as a suppressed
     * exception. If closing fails after successful processing, iteration
     * stops and the close failure is propagated. No later batch is requested
     * after either failure.</p>
     *
     * @param batches lazy source of resource-owning batches
     * @param processor operation to perform for each batch
     * @param <B> batch type
     * @return number of batches successfully processed and closed
     * @throws Exception if processing or closing a batch fails
     */
    public static <B extends AutoCloseable> long consume(
            Iterable<? extends B> batches,
            BatchProcessor<? super B> processor) throws Exception {
        Objects.requireNonNull(batches, "batches");
        Objects.requireNonNull(processor, "processor");

        long processed = 0;
        for (B batch : batches) {
            try (B ownedBatch = Objects.requireNonNull(batch, "batch")) {
                processor.process(ownedBatch);
            }
            processed = Math.incrementExact(processed);
        }
        return processed;
    }

    /** A batch operation that may fail with a checked exception. */
    @FunctionalInterface
    public interface BatchProcessor<B> {
        void process(B batch) throws Exception;
    }
}
