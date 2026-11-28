package tech.kayys.aljabr.diffusion.video;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;

/**
 * Validated, ordered view over the prepared frames belonging to one video input.
 *
 * <p>This type does not take ownership of frame data. Its lifetime is bounded
 * by the {@link PreparedDiffusionConditioningSet} (or other owner) that owns
 * the supplied {@link VideoFrameConditioningData} instances. Closing that
 * owner invalidates any resources exposed by the frames.</p>
 *
 * <p>Frames must already be supplied in strictly increasing sequence-index
 * and presentation-timestamp order. Sequence-index gaps are allowed because
 * temporal sampling may intentionally be sparse. The sequence never sorts,
 * resamples, interpolates, or invents timestamps.</p>
 */
public final class VideoFrameSequence {
    private final List<VideoFrameConditioningData> frames;
    private final Duration duration;

    private VideoFrameSequence(List<VideoFrameConditioningData> frames) {
        this.frames = List.copyOf(frames);
        Instant first = this.frames.get(0).timestamp();
        Instant last = this.frames.get(this.frames.size() - 1).timestamp();
        this.duration = Duration.between(first, last);
    }

    /**
     * Creates a non-owning sequence view from an already prepared conditioning set.
     * Non-video conditioning entries are ignored; every VIDEO_FRAME entry must
     * have been prepared by the video-frame adapter.
     */
    public static VideoFrameSequence from(PreparedDiffusionConditioningSet prepared) {
        Objects.requireNonNull(prepared, "prepared");
        List<VideoFrameConditioningData> frames = new ArrayList<>();
        prepared.data().stream()
                .filter(data -> data.modality()
                        == tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput.Modality.VIDEO_FRAME)
                .forEach(data -> {
                    if (!(data instanceof VideoFrameConditioningData frame)) {
                        throw new IllegalArgumentException(
                                "VIDEO_FRAME conditioning data must be VideoFrameConditioningData, got "
                                        + data.getClass().getName());
                    }
                    frames.add(frame);
                });
        return of(frames);
    }

    /**
     * Creates a non-owning sequence view from frame data in caller-declared order.
     *
     * @throws IllegalArgumentException if the sequence is empty, indices are
     *         not strictly increasing, or timestamps are not strictly increasing
     */
    public static VideoFrameSequence of(List<VideoFrameConditioningData> frames) {
        Objects.requireNonNull(frames, "frames");
        List<VideoFrameConditioningData> copy = List.copyOf(frames);
        if (copy.isEmpty()) {
            throw new IllegalArgumentException("video frame sequence must contain at least one frame");
        }

        VideoFrameConditioningData previous = null;
        for (VideoFrameConditioningData current : copy) {
            Objects.requireNonNull(current, "frame");
            if (previous != null) {
                if (current.sequenceIndex() <= previous.sequenceIndex()) {
                    throw new IllegalArgumentException(
                            "video frame sequence indices must be strictly increasing: "
                                    + previous.sequenceIndex() + " then " + current.sequenceIndex());
                }
                if (!current.timestamp().isAfter(previous.timestamp())) {
                    throw new IllegalArgumentException(
                            "video frame timestamps must be strictly increasing: "
                                    + previous.timestamp() + " then " + current.timestamp());
                }
            }
            previous = current;
        }
        return new VideoFrameSequence(copy);
    }

    /** Frames in validated temporal order. The returned list is immutable. */
    public List<VideoFrameConditioningData> frames() {
        return frames;
    }

    public int frameCount() {
        return frames.size();
    }

    public VideoFrameConditioningData firstFrame() {
        return frames.get(0);
    }

    public VideoFrameConditioningData lastFrame() {
        return frames.get(frames.size() - 1);
    }

    /** Elapsed presentation time from the first frame to the last frame. */
    public Duration duration() {
        return duration;
    }

    /** Finds a frame by its original sequence index without assuming contiguous indices. */
    public Optional<VideoFrameConditioningData> frameAtSequenceIndex(int sequenceIndex) {
        return frames.stream()
                .filter(frame -> frame.sequenceIndex() == sequenceIndex)
                .findFirst();
    }
}
