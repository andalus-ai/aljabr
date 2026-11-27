package tech.kayys.aljabr.diffusion.image;

import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreprocessor;

/** Reusable image-conditioning adapter with injected decoding and transform capabilities. */
public final class ImageConditioningPreprocessor implements DiffusionConditioningPreprocessor {
    private static final Set<DiffusionConditioningInput.Modality> MODALITIES =
            Set.of(DiffusionConditioningInput.Modality.IMAGE,
                    DiffusionConditioningInput.Modality.REFERENCE_IMAGE);
    private final ImageDecoder decoder;
    private final ImageConditioningTransform transform;

    public ImageConditioningPreprocessor(ImageDecoder decoder) {
        this(decoder, ImageConditioningTransform.identity());
    }
    public ImageConditioningPreprocessor(ImageDecoder decoder, ImageConditioningTransform transform) {
        this.decoder = Objects.requireNonNull(decoder, "decoder");
        this.transform = Objects.requireNonNull(transform, "transform");
    }
    @Override public String id() { return "aljabr.image.conditioning"; }
    @Override public Set<DiffusionConditioningInput.Modality> modalities() { return MODALITIES; }

    @Override
    public DiffusionConditioningData prepare(DiffusionConditioningInput input,
            DiffusionConditioningPreparationContext context) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(context, "context");
        if (!supports(input)) {
            throw new DiffusionConditioningPreparationException(
                    input, "Image preprocessor cannot prepare modality " + input.modality());
        }
        try {
            DecodedImage decoded = Objects.requireNonNull(decoder.decode(input), "image decoder returned null");
            DecodedImage prepared = Objects.requireNonNull(
                    transform.transform(input, decoded, context), "image transform returned null");
            return new ImageConditioningData(input.id(), input.modality(), prepared);
        } catch (Exception failure) {
            if (failure instanceof DiffusionConditioningPreparationException preparationFailure) {
                throw preparationFailure;
            }
            throw new DiffusionConditioningPreparationException(
                    input, "Failed to prepare image input '" + input.id() + "'", failure);
        }
    }
}
