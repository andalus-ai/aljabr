package tech.kayys.aljabr.diffusion.video;

import java.util.Iterator;
import java.util.Objects;
import java.util.function.Function;

/**
 * Lazily maps bounded video-frame chunks into caller-defined batch values.
 *
 * <p>This adapter deliberately does not depend on a concrete packed-batch
 * representation. The caller supplies the packing function, which may use
 * Aljabr's packed frame APIs or an application-specific representation.</p>
 *
 * <p>Each iterator requests one source chunk and maps it only when
 * {@link Iterator#next()} is called. The iterable retains neither produced
 * values nor prior chunks. Produced values are transferred to the consumer;
 * if a value owns resources (for example, implements {@link AutoCloseable}),
 * the consumer is responsible for releasing it.</p>
 *
 * <p>The source sequence is a non-owning view. Its original conditioning
 * owner must remain open for as long as the iterator may request chunks or
 * the supplied mapper may read frame data.</p>
 */
public final class VideoFrameBatchStream {
    private VideoFrameBatchStream() {
    }

    /**
     * Creates a repeatable, pull-based iterable of packed batch values.
     *
     * @param sequence source video sequence
     * @param chunkSize maximum number of frames passed to the packer per value
     * @param tailPolicy policy for a final incomplete chunk
     * @param batchPacker maps each chunk to a caller-owned batch value
     * @param <B> caller-defined batch representation
     * @return a lazy iterable; no chunks or batch values are materialized up front
     */
    public static <B> Iterable<B> mapChunks(
            VideoFrameSequence sequence,
            int chunkSize,
            VideoFrameSequenceChunker.TailPolicy tailPolicy,
            Function<? super VideoFrameSequence, ? extends B> batchPacker) {
        Objects.requireNonNull(batchPacker, "batchPacker");
        Iterable<VideoFrameSequence> chunks =
                VideoFrameSequenceChunker.chunks(sequence, chunkSize, tailPolicy);

        return () -> {
            Iterator<VideoFrameSequence> source = chunks.iterator();
            return new Iterator<>() {
                @Override
                public boolean hasNext() {
                    return source.hasNext();
                }

                @Override
                public B next() {
                    VideoFrameSequence chunk = source.next();
                    return Objects.requireNonNull(
                            batchPacker.apply(chunk),
                            "batchPacker must not return null");
                }
            };
        };
    }
}
