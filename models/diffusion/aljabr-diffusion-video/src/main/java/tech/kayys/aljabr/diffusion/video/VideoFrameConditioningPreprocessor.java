package tech.kayys.aljabr.diffusion.video;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreprocessor;
/** Prepares one video frame while preserving its caller-supplied order and presentation timestamp. */
public final class VideoFrameConditioningPreprocessor implements DiffusionConditioningPreprocessor {
    public static final String TIMESTAMP_METADATA_KEY = "timestamp";
    private static final Set<DiffusionConditioningInput.Modality> MODALITIES = Set.of(DiffusionConditioningInput.Modality.VIDEO_FRAME);
    private final VideoFrameDecoder decoder;
    private final VideoFrameConditioningTransform transform;
    public VideoFrameConditioningPreprocessor(VideoFrameDecoder decoder) { this(decoder, VideoFrameConditioningTransform.identity()); }
    public VideoFrameConditioningPreprocessor(VideoFrameDecoder decoder, VideoFrameConditioningTransform transform) {
        this.decoder = Objects.requireNonNull(decoder, "decoder");
        this.transform = Objects.requireNonNull(transform, "transform");
    }
    @Override public String id() { return "aljabr.video.frame.conditioning"; }
    @Override public Set<DiffusionConditioningInput.Modality> modalities() { return MODALITIES; }
    @Override public boolean supports(DiffusionConditioningInput input) {
        return DiffusionConditioningPreprocessor.super.supports(input) && input.metadata().containsKey(TIMESTAMP_METADATA_KEY);
    }
    @Override public DiffusionConditioningData prepare(DiffusionConditioningInput input, DiffusionConditioningPreparationContext context) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(context, "context");
        if (!supports(input)) throw new DiffusionConditioningPreparationException(input,
                "Video frame adapter requires VIDEO_FRAME modality and ISO-8601 timestamp metadata");
        try {
            Instant timestamp = Instant.parse(input.metadata().get(TIMESTAMP_METADATA_KEY));
            DecodedVideoFrame decoded = Objects.requireNonNull(decoder.decode(input), "video frame decoder returned null");
            DecodedVideoFrame prepared = Objects.requireNonNull(transform.transform(input, decoded, context), "video frame transform returned null");
            return new VideoFrameConditioningData(input.id(), input.sequenceIndex(), timestamp, prepared);
        } catch (DateTimeParseException failure) {
            throw new DiffusionConditioningPreparationException(input, "Invalid ISO-8601 frame timestamp for input '" + input.id() + "'", failure);
        } catch (Exception failure) {
            if (failure instanceof DiffusionConditioningPreparationException e) throw e;
            throw new DiffusionConditioningPreparationException(input, "Failed to prepare video frame input '" + input.id() + "'", failure);
        }
    }
}
