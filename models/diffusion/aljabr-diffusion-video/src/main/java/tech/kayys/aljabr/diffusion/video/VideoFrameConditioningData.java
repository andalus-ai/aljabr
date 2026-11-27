package tech.kayys.aljabr.diffusion.video;
import java.time.Instant;
import java.util.Objects;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
/** Caller-owned immutable pixels and temporal provenance for one video frame. */
public final class VideoFrameConditioningData implements DiffusionConditioningData {
    private final String inputId;
    private final int sequenceIndex;
    private final Instant timestamp;
    private final DecodedVideoFrame frame;
    public VideoFrameConditioningData(String inputId, int sequenceIndex, Instant timestamp, DecodedVideoFrame frame) {
        this.inputId = Objects.requireNonNull(inputId, "inputId");
        if (inputId.isBlank()) throw new IllegalArgumentException("inputId must not be blank");
        if (sequenceIndex < 0) throw new IllegalArgumentException("sequenceIndex must be non-negative");
        this.sequenceIndex = sequenceIndex;
        this.timestamp = Objects.requireNonNull(timestamp, "timestamp");
        this.frame = Objects.requireNonNull(frame, "frame");
    }
    @Override public String inputId() { return inputId; }
    @Override public DiffusionConditioningInput.Modality modality() { return DiffusionConditioningInput.Modality.VIDEO_FRAME; }
    public int sequenceIndex() { return sequenceIndex; }
    public Instant timestamp() { return timestamp; }
    public int width() { return frame.width(); }
    public int height() { return frame.height(); }
    public int channels() { return frame.channels(); }
    public DecodedVideoFrame.Layout layout() { return frame.layout(); }
    public float[] pixels() { return frame.pixels(); }
    @Override public void close() { /* Heap-owned frame pixels. */ }
}
