package tech.kayys.aljabr.diffusion.mask;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;

/**
 * Optional mask-aware crop/resize/transform boundary.
 * Implementations must honor the supplied resampling policy: nearest-neighbor
 * for binary labels and bilinear interpolation for soft weights.
 */
@FunctionalInterface
public interface MaskConditioningTransform {
    DecodedMask transform(DiffusionConditioningInput input, DecodedMask decoded,
            MaskConditioningOptions options, DiffusionConditioningPreparationContext context) throws Exception;

    static MaskConditioningTransform identity() {
        return (input, decoded, options, context) -> decoded;
    }
}
