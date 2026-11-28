package tech.kayys.aljabr.diffusion.video;

import java.time.DateTimeException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Keeps the first frame, then the first available frame at or after each
 * interval target measured from the first frame's timestamp.
 */
public final class TimestampIntervalVideoFrameSamplingPolicy implements VideoFrameSamplingPolicy {
    private final Duration interval;

    public TimestampIntervalVideoFrameSamplingPolicy(Duration interval) {
        this.interval = Objects.requireNonNull(interval, "interval");
        if (interval.isZero() || interval.isNegative()) {
            throw new IllegalArgumentException("interval must be positive");
        }
    }

    public Duration interval() {
        return interval;
    }

    @Override
    public List<VideoFrameConditioningData> select(VideoFrameSequence sequence) {
        List<VideoFrameConditioningData> frames = sequence.frames();
        List<VideoFrameConditioningData> selected = new ArrayList<>();
        selected.add(frames.get(0));

        Instant target;
        try {
            target = frames.get(0).timestamp().plus(interval);
        } catch (DateTimeException | ArithmeticException overflow) {
            return List.copyOf(selected);
        }

        int cursor = 1;
        while (cursor < frames.size()) {
            VideoFrameConditioningData candidate = null;
            while (cursor < frames.size()) {
                VideoFrameConditioningData frame = frames.get(cursor++);
                if (!frame.timestamp().isBefore(target)) {
                    candidate = frame;
                    break;
                }
            }
            if (candidate == null) {
                break;
            }
            selected.add(candidate);
            // Skip cadence targets already passed by this selected frame. This
            // prevents a large timestamp gap from degenerating into every-frame selection.
            try {
                do {
                    target = target.plus(interval);
                } while (!target.isAfter(candidate.timestamp()));
            } catch (DateTimeException | ArithmeticException overflow) {
                break;
            }
        }
        return List.copyOf(selected);
    }
}
