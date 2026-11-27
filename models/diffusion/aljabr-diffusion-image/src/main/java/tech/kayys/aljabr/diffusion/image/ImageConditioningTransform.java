package tech.kayys.aljabr.diffusion.image;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;

/** Optional resize/crop/channel/normalization boundary, independent of image libraries and tensor backends. */
@FunctionalInterface
public interface ImageConditioningTransform {
    DecodedImage transform(DiffusionConditioningInput input, DecodedImage decoded,
            DiffusionConditioningPreparationContext context) throws Exception;

    static ImageConditioningTransform identity() {
        return (input, decoded, context) -> decoded;
    }
}
