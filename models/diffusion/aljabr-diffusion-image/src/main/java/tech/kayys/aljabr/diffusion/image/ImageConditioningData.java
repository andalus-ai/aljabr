package tech.kayys.aljabr.diffusion.image;

import java.util.Objects;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;

/** Caller-owned immutable image-conditioning pixels and explicit shape. */
public final class ImageConditioningData implements DiffusionConditioningData {
    private final String inputId;
    private final DiffusionConditioningInput.Modality modality;
    private final DecodedImage image;

    public ImageConditioningData(String inputId, DiffusionConditioningInput.Modality modality, DecodedImage image) {
        this.inputId = Objects.requireNonNull(inputId, "inputId");
        if (inputId.isBlank()) throw new IllegalArgumentException("inputId must not be blank");
        this.modality = Objects.requireNonNull(modality, "modality");
        if (modality != DiffusionConditioningInput.Modality.IMAGE
                && modality != DiffusionConditioningInput.Modality.REFERENCE_IMAGE) {
            throw new IllegalArgumentException("image data requires IMAGE or REFERENCE_IMAGE modality");
        }
        this.image = Objects.requireNonNull(image, "image");
    }
    @Override public String inputId() { return inputId; }
    @Override public DiffusionConditioningInput.Modality modality() { return modality; }
    public int width() { return image.width(); }
    public int height() { return image.height(); }
    public int channels() { return image.channels(); }
    public DecodedImage.Layout layout() { return image.layout(); }
    public float[] pixels() { return image.pixels(); }
    @Override public void close() { /* Heap-owned pixel storage. */ }
}
