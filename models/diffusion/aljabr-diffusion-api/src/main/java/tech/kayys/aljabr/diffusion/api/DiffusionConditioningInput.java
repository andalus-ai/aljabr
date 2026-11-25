package tech.kayys.aljabr.diffusion.api;

import java.util.Map;
import java.util.Objects;

/**
 * Describes one caller-supplied conditioning input without prescribing how it
 * is decoded, embedded, transferred to a device, or consumed by a model.
 *
 * <p>{@code source} is opaque to this API. Applications may use a URI, an
 * application-managed resource key, or another stable source reference.</p>
 */
public record DiffusionConditioningInput(
        String id,
        Modality modality,
        String source,
        Integer sequenceIndex,
        Map<String, String> metadata) {

    public DiffusionConditioningInput {
        id = requireText(id, "id");
        Objects.requireNonNull(modality, "modality");
        source = requireText(source, "source");
        metadata = Map.copyOf(Objects.requireNonNull(metadata, "metadata"));
        metadata.forEach((key, value) -> {
            requireText(key, "metadata key");
            Objects.requireNonNull(value, "metadata value");
        });

        if (modality == Modality.VIDEO_FRAME) {
            if (sequenceIndex == null || sequenceIndex < 0) {
                throw new IllegalArgumentException("video frames require a non-negative sequenceIndex");
            }
        } else if (sequenceIndex != null) {
            throw new IllegalArgumentException("sequenceIndex is only valid for VIDEO_FRAME inputs");
        }
    }

    public DiffusionConditioningInput(String id, Modality modality, String source) {
        this(id, modality, source, null, Map.of());
    }

    public DiffusionConditioningInput(String id, String source, int sequenceIndex) {
        this(id, Modality.VIDEO_FRAME, source, sequenceIndex, Map.of());
    }

    public enum Modality {
        PROMPT,
        IMAGE,
        REFERENCE_IMAGE,
        VIDEO_FRAME,
        MASK
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
