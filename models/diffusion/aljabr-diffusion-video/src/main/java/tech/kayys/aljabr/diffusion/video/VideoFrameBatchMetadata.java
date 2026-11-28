package tech.kayys.aljabr.diffusion.video;

import java.time.Instant;
import java.util.Objects;

/** Original temporal and source provenance for one position in a packed batch. */
public record VideoFrameBatchMetadata(String inputId, int sequenceIndex, Instant timestamp) {
    public VideoFrameBatchMetadata {
        Objects.requireNonNull(inputId, "inputId");
        Objects.requireNonNull(timestamp, "timestamp");
        if (inputId.isBlank()) {
            throw new IllegalArgumentException("inputId must not be blank");
        }
        if (sequenceIndex < 0) {
            throw new IllegalArgumentException("sequenceIndex must be non-negative");
        }
    }

    static VideoFrameBatchMetadata from(VideoFrameConditioningData frame) {
        return new VideoFrameBatchMetadata(frame.inputId(), frame.sequenceIndex(), frame.timestamp());
    }
}
