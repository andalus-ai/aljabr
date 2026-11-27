package tech.kayys.aljabr.diffusion.mask;

import java.util.Objects;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningData;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;

/** Caller-owned immutable mask data with explicit binary/soft semantics. */
public final class MaskConditioningData implements DiffusionConditioningData {
    private final String inputId;
    private final DecodedMask mask;
    private final MaskSemantics semantics;

    public MaskConditioningData(String inputId, DecodedMask mask, MaskSemantics semantics) {
        this.inputId = Objects.requireNonNull(inputId, "inputId");
        if (inputId.isBlank()) throw new IllegalArgumentException("inputId must not be blank");
        this.mask = Objects.requireNonNull(mask, "mask");
        this.semantics = Objects.requireNonNull(semantics, "semantics");
        if (semantics == MaskSemantics.BINARY) {
            for (float value : mask.values()) {
                if (value != 0f && value != 1f) {
                    throw new IllegalArgumentException("binary mask values must be exactly zero or one");
                }
            }
        }
    }
    @Override public String inputId() { return inputId; }
    @Override public DiffusionConditioningInput.Modality modality() { return DiffusionConditioningInput.Modality.MASK; }
    public int width() { return mask.width(); }
    public int height() { return mask.height(); }
    public MaskSemantics semantics() { return semantics; }
    public float[] values() { return mask.values(); }
    @Override public void close() { /* Heap-owned mask storage. */ }
}
