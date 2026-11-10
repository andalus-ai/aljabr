package tech.kayys.aljabr.spi.provider;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/** Immutable metadata for an independently deployable Aljabr provider. */
public record ProviderDescriptor(String id, String version, String apiCompatibility,
                                 Set<CapabilityDescriptor> capabilities) {
    public ProviderDescriptor {
        id = requireText(id, "id");
        if (!id.matches("[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*")) {
            throw new IllegalArgumentException("id must use lowercase namespace/name form: " + id);
        }
        version = requireText(version, "version");
        apiCompatibility = requireText(apiCompatibility, "apiCompatibility");
        ApiCompatibilityRange.parse(apiCompatibility);
        capabilities = Set.copyOf(Objects.requireNonNull(capabilities, "capabilities"));
        Set<String> capabilityIds = new HashSet<>();
        for (CapabilityDescriptor capability : capabilities) {
            if (!capabilityIds.add(capability.id())) {
                throw new IllegalArgumentException("duplicate capability id: " + capability.id());
            }
        }
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }
}
