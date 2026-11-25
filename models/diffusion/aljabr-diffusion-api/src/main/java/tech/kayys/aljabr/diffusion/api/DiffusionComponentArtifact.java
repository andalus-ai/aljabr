package tech.kayys.aljabr.diffusion.api;

import java.util.Map;
import java.util.Objects;

/**
 * Logical reference to one independently loadable diffusion-model component.
 *
 * <p>The role is a stable, family-neutral identifier such as
 * {@code diffusion-model}, {@code text-encoder}, or {@code video-autoencoder}.
 * The location is opaque to this API: storage adapters decide how to resolve
 * local paths, repository files, object-store URIs, or other locations.</p>
 */
public record DiffusionComponentArtifact(
        String id,
        String role,
        String location,
        String format,
        String variant,
        String sha256,
        boolean required,
        Map<String, String> metadata) {

    public DiffusionComponentArtifact {
        id = requireText(id, "id");
        role = requireText(role, "role");
        location = requireText(location, "location");
        format = requireText(format, "format");
        variant = normalizeOptional(variant);
        sha256 = normalizeOptional(sha256);
        if (sha256 != null && !sha256.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalArgumentException("sha256 must contain exactly 64 hexadecimal characters");
        }
        metadata = Map.copyOf(Objects.requireNonNull(metadata, "metadata"));
        metadata.forEach((key, value) -> {
            requireText(key, "metadata key");
            Objects.requireNonNull(value, "metadata value");
        });
    }

    public DiffusionComponentArtifact(
            String id, String role, String location, String format, boolean required) {
        this(id, role, location, format, null, null, required, Map.of());
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }

    private static String normalizeOptional(String value) {
        return value == null || value.isBlank() ? null : value;
    }
}
