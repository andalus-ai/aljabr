package tech.kayys.aljabr.diffusion.api;

import java.util.Objects;
import java.util.Set;

/**
 * A capability's requirement for a logical model component role.
 *
 * <p>An empty acceptedFormats set accepts any artifact format. A preferred
 * variant is a preference only; it never makes an otherwise compatible
 * artifact invalid.</p>
 */
public record DiffusionComponentRequirement(
        String role,
        boolean required,
        Set<String> acceptedFormats,
        String preferredVariant) {

    public DiffusionComponentRequirement {
        role = requireText(role, "role");
        acceptedFormats = Set.copyOf(Objects.requireNonNull(acceptedFormats, "acceptedFormats"));
        acceptedFormats.forEach(format -> requireText(format, "accepted format"));
        preferredVariant = preferredVariant == null || preferredVariant.isBlank()
                ? null : preferredVariant;
    }

    public DiffusionComponentRequirement(String role, boolean required) {
        this(role, required, Set.of(), null);
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }
}
