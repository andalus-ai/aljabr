package tech.kayys.aljabr.spi.provider;

import java.util.Objects;

/**
 * Provider-owned declaration of one Aljabr capability.
 * IDs use lowercase dotted segments; each segment starts with a letter or digit
 * and may contain lowercase letters, digits and hyphens.
 */
public record CapabilityDescriptor(String id, String version, CapabilityStability stability) {
    public CapabilityDescriptor {
        id = requireText(id, "id");
        if (!id.matches("[a-z0-9]+(?:-[a-z0-9]+)*(?:\\.[a-z0-9]+(?:-[a-z0-9]+)*)*")) {
            throw new IllegalArgumentException("id must use lowercase dotted capability segments: " + id);
        }
        version = requireText(version, "version");
        Objects.requireNonNull(stability, "stability");
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }
}
