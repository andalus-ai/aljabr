package tech.kayys.aljabr.diffusion.video;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/** Applies a sampling policy and validates its result as a non-owning sequence view. */
public final class VideoFrameSampler {
    private VideoFrameSampler() {
    }

    /**
     * Returns an empty result when the policy selects no frames. Selected frames
     * must be original instances from the input sequence, in strictly increasing
     * source order, with no duplicates.
     */
    public static Optional<VideoFrameSequence> sample(
            VideoFrameSequence sequence, VideoFrameSamplingPolicy policy) {
        Objects.requireNonNull(sequence, "sequence");
        Objects.requireNonNull(policy, "policy");

        List<VideoFrameConditioningData> selected =
                List.copyOf(Objects.requireNonNull(policy.select(sequence), "policy result"));
        if (selected.isEmpty()) {
            return Optional.empty();
        }

        IdentityHashMap<VideoFrameConditioningData, Integer> sourcePositions = new IdentityHashMap<>();
        List<VideoFrameConditioningData> source = sequence.frames();
        for (int i = 0; i < source.size(); i++) {
            sourcePositions.put(source.get(i), i);
        }

        int previousPosition = -1;
        for (VideoFrameConditioningData frame : selected) {
            Objects.requireNonNull(frame, "selected frame");
            Integer position = sourcePositions.get(frame);
            if (position == null) {
                throw new IllegalArgumentException("sampling policy returned a frame outside the source sequence");
            }
            if (position <= previousPosition) {
                throw new IllegalArgumentException(
                        "sampling policy must preserve source order and must not return duplicate frames");
            }
            previousPosition = position;
        }
        return Optional.of(VideoFrameSequence.of(selected));
    }
}
