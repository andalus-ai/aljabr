package tech.kayys.aljabr.diffusion.mask;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;

/** Decodes an opaque mask source into normalized single-channel values. */
@FunctionalInterface
public interface MaskDecoder {
    DecodedMask decode(DiffusionConditioningInput input) throws Exception;
}
