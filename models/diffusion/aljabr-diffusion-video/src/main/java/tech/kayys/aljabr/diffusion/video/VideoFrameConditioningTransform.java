package tech.kayys.aljabr.diffusion.video;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
/** Applies frame preprocessing without selecting a tensor backend or device. */
@FunctionalInterface
public interface VideoFrameConditioningTransform {
    DecodedVideoFrame transform(DiffusionConditioningInput input, DecodedVideoFrame decoded,
            DiffusionConditioningPreparationContext context) throws Exception;
    static VideoFrameConditioningTransform identity() { return (input, decoded, context) -> decoded; }
}
