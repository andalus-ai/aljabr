package tech.kayys.aljabr.diffusion.video;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
/** Decodes one opaque VIDEO_FRAME input; codec and source concerns remain injected. */
@FunctionalInterface
public interface VideoFrameDecoder {
    DecodedVideoFrame decode(DiffusionConditioningInput input) throws Exception;
}
