package tech.kayys.aljabr.spi.provider;
import java.util.Objects;
/** Provider-owned declaration of one Aljabr capability. */
public record CapabilityDescriptor(String id, String version, CapabilityStability stability) {
    public CapabilityDescriptor {
        id = requireText(id, "id");
        version = requireText(version, "version");
        Objects.requireNonNull(stability, "stability");
    }
    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value.trim();
    }
}
