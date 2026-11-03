package tech.kayys.aljabr.spi.provider;
import java.util.Objects;
import java.util.Set;
/** Immutable metadata for an independently deployable Aljabr provider. */
public record ProviderDescriptor(String id, String version, String apiCompatibility,
                                 Set<CapabilityDescriptor> capabilities) {
    public ProviderDescriptor {
        id = requireText(id, "id");
        if (!id.matches("[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*"))
            throw new IllegalArgumentException("id must use namespace/name form: " + id);
        version = requireText(version, "version");
        apiCompatibility = requireText(apiCompatibility, "apiCompatibility");
        capabilities = Set.copyOf(Objects.requireNonNull(capabilities, "capabilities"));
    }
    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value.trim();
    }
}
