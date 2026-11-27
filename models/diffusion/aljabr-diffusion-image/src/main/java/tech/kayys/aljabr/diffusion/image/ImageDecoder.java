package tech.kayys.aljabr.diffusion.image;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;

/** Decodes an opaque conditioning source; implementations may use ImageIO or another optional decoder. */
@FunctionalInterface
public interface ImageDecoder {
    DecodedImage decode(DiffusionConditioningInput input) throws Exception;
}
