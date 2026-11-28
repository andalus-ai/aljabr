package tech.kayys.aljabr.diffusion.video;

import java.time.Instant;
import java.util.List;
import java.util.Objects;

/** Selects frames within an inclusive presentation-timestamp window. */
public final class TimestampWindowVideoFrameSamplingPolicy implements VideoFrameSamplingPolicy {
    private final Instant startInclusive;
    private final Instant endInclusive;

    public TimestampWindowVideoFrameSamplingPolicy(Instant startInclusive, Instant endInclusive) {
        this.startInclusive = Objects.requireNonNull(startInclusive, "startInclusive");
        this.endInclusive = Objects.requireNonNull(endInclusive, "endInclusive");
        if (endInclusive.isBefore(startInclusive)) {
            throw new IllegalArgumentException("endInclusive must not be before startInclusive");
        }
    }

    public Instant startInclusive() {
        return startInclusive;
    }

    public Instant endInclusive() {
        return endInclusive;
    }

    @Override
    public List<VideoFrameConditioningData> select(VideoFrameSequence sequence) {
        return sequence.frames().stream()
                .filter(frame -> !frame.timestamp().isBefore(startInclusive))
                .filter(frame -> !frame.timestamp().isAfter(endInclusive))
                .toList();
    }
}
