package tech.kayys.aljabr.diffusion.video;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * Lazily exposes a sequence as contiguous, bounded frame chunks.
 *
 * <p>Each iterator creates non-owning {@link VideoFrameSequence} views on
 * demand. No list of all chunks is materialized, so a consumer controls
 * progression by requesting the next chunk. This is pull-based flow control,
 * not an asynchronous scheduler or a background producer.</p>
 */
public final class VideoFrameSequenceChunker {
    private VideoFrameSequenceChunker() {
    }

    /**
     * Creates a repeatable, lazy iterable of contiguous chunks.
     *
     * @param sequence source sequence
     * @param chunkSize maximum frames per chunk; must be positive
     * @param tailPolicy whether to include a final incomplete chunk
     * @return an iterable whose iterators independently traverse the source
     */
    public static Iterable<VideoFrameSequence> chunks(
            VideoFrameSequence sequence,
            int chunkSize,
            TailPolicy tailPolicy) {
        Objects.requireNonNull(sequence, "sequence");
        Objects.requireNonNull(tailPolicy, "tailPolicy");
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be positive");
        }

        return () -> new Iterator<>() {
            private int nextStart;

            @Override
            public boolean hasNext() {
                int remaining = sequence.frameCount() - nextStart;
                return remaining > 0
                        && (tailPolicy == TailPolicy.INCLUDE_PARTIAL || remaining >= chunkSize);
            }

            @Override
            public VideoFrameSequence next() {
                if (!hasNext()) {
                    throw new NoSuchElementException("no video frame chunks remain");
                }
                int end = (int) Math.min((long) nextStart + chunkSize, sequence.frameCount());
                VideoFrameSequence chunk = VideoFrameSequence.of(
                        sequence.frames().subList(nextStart, end));
                nextStart = end;
                return chunk;
            }
        };
    }

    /** Controls whether a final chunk smaller than the configured size is retained. */
    public enum TailPolicy {
        /** Retain the final non-empty chunk even when it contains fewer frames. */
        INCLUDE_PARTIAL,
        /** Emit only chunks containing exactly chunkSize frames. */
        DROP_PARTIAL
    }
}
