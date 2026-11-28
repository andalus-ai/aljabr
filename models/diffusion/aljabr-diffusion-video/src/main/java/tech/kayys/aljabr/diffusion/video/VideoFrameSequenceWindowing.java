package tech.kayys.aljabr.diffusion.video;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Creates bounded, non-owning temporal windows over a prepared video sequence.
 *
 * <p>Windowing changes only the grouping of frames. It never copies pixels,
 * changes timestamps or sequence indices, sorts frames, or takes ownership.</p>
 */
public final class VideoFrameSequenceWindowing {
    private VideoFrameSequenceWindowing() {
    }

    /**
     * Splits a sequence into windows using a fixed frame count and start stride.
     *
     * @param sequence source sequence
     * @param windowSize maximum frames in each window; must be positive
     * @param stride distance between successive window starts; must be positive
     * @param tailPolicy whether to retain a final window smaller than windowSize
     * @return immutable windows in source order; may be empty with DROP_PARTIAL
     */
    public static List<VideoFrameSequence> window(
            VideoFrameSequence sequence,
            int windowSize,
            int stride,
            TailPolicy tailPolicy) {
        Objects.requireNonNull(sequence, "sequence");
        Objects.requireNonNull(tailPolicy, "tailPolicy");
        if (windowSize <= 0) {
            throw new IllegalArgumentException("windowSize must be positive");
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("stride must be positive");
        }

        int frameCount = sequence.frameCount();
        List<VideoFrameSequence> windows = new ArrayList<>();
        for (long start = 0; start < frameCount; start += stride) {
            long end = Math.min(start + windowSize, frameCount);
            boolean complete = end - start == windowSize;
            if (!complete && tailPolicy == TailPolicy.DROP_PARTIAL) {
                break;
            }
            windows.add(VideoFrameSequence.of(
                    sequence.frames().subList((int) start, (int) end)));
        }
        return List.copyOf(windows);
    }

    /** Controls whether a final incomplete window is retained. */
    public enum TailPolicy {
        /** Retain every non-empty trailing window, including a short overlap tail. */
        INCLUDE_PARTIAL,
        /** Emit only windows containing exactly windowSize frames. */
        DROP_PARTIAL
    }
}
