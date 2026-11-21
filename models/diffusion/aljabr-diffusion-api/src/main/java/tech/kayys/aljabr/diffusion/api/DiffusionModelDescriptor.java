package tech.kayys.aljabr.diffusion.api;

import java.util.Map;
import java.util.Objects;

/**
 * Family-neutral description of a diffusion model artifact.
 *
 * <p>Family modules may define additional metadata keys, but the descriptor
 * deliberately does not encode checkpoint-format or backend-specific types.</p>
 */
public record DiffusionModelDescriptor(
        String familyId,
        String architecture,
        Map<String, String> attributes) {

    public DiffusionModelDescriptor {
        familyId = requireText(familyId, "familyId");
        architecture = requireText(architecture, "architecture");
        attributes = Map.copyOf(Objects.requireNonNull(attributes, "attributes"));
        attributes.forEach((key, value) -> {
            requireText(key, "attribute key");
            Objects.requireNonNull(value, "attribute value");
        });
    }

    public DiffusionModelDescriptor(String familyId, String architecture) {
        this(familyId, architecture, Map.of());
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
