package tech.kayys.aljabr.diffusion.mask;

import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreprocessor;

/** Reusable mask adapter; binary thresholding and resampling semantics are explicit. */
public final class MaskConditioningPreprocessor implements DiffusionConditioningPreprocessor {
    private static final Set<DiffusionConditioningInput.Modality> MODALITIES =
            Set.of(DiffusionConditioningInput.Modality.MASK);
    private final MaskDecoder decoder;
    private final MaskConditioningTransform transform;
    private final MaskConditioningOptions options;

    public MaskConditioningPreprocessor(MaskDecoder decoder, MaskConditioningOptions options) {
        this(decoder, MaskConditioningTransform.identity(), options);
    }
    public MaskConditioningPreprocessor(MaskDecoder decoder, MaskConditioningTransform transform,
            MaskConditioningOptions options) {
        this.decoder = Objects.requireNonNull(decoder, "decoder");
        this.transform = Objects.requireNonNull(transform, "transform");
        this.options = Objects.requireNonNull(options, "options");
    }
    @Override public String id() { return "aljabr.mask.conditioning"; }
    @Override public Set<DiffusionConditioningInput.Modality> modalities() { return MODALITIES; }

    @Override
    public DiffusionConditioningData prepare(DiffusionConditioningInput input,
            DiffusionConditioningPreparationContext context) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(context, "context");
        if (!supports(input)) {
            throw new DiffusionConditioningPreparationException(
                    input, "Mask preprocessor cannot prepare modality " + input.modality());
        }
        try {
            DecodedMask decoded = Objects.requireNonNull(decoder.decode(input), "mask decoder returned null");
            DecodedMask transformed = Objects.requireNonNull(
                    transform.transform(input, decoded, options, context), "mask transform returned null");
            if (options.semantics() == MaskSemantics.SOFT) {
                return new MaskConditioningData(input.id(), transformed, MaskSemantics.SOFT);
            }
            float[] values = transformed.values();
            for (int i = 0; i < values.length; i++) {
                values[i] = values[i] >= options.threshold() ? 1f : 0f;
            }
            return new MaskConditioningData(input.id(),
                    new DecodedMask(transformed.width(), transformed.height(), values), MaskSemantics.BINARY);
        } catch (Exception failure) {
            if (failure instanceof DiffusionConditioningPreparationException preparationFailure) {
                throw preparationFailure;
            }
            throw new DiffusionConditioningPreparationException(
                    input, "Failed to prepare mask input '" + input.id() + "'", failure);
        }
    }
}
