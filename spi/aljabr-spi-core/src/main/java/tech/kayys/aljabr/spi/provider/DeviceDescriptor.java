package tech.kayys.aljabr.spi.provider;

import java.util.Objects;

/**
 * Immutable snapshot identifying a device within one provider.
 * Device ids are provider-local and must not be treated as globally unique.
 * Dynamic telemetry such as utilization and temperature belongs elsewhere.
 */
public record DeviceDescriptor(
        String providerId,
        String deviceId,
        String name,
        DeviceKind kind,
        Long totalMemoryBytes,
        Long freeMemoryBytes,
        String vendor,
        String architecture,
        String driverVersion,
        String runtimeVersion) {

    public DeviceDescriptor {
        providerId = requireText(providerId, "providerId");
        deviceId = requireText(deviceId, "deviceId");
        name = requireText(name, "name");
        Objects.requireNonNull(kind, "kind");
        if (totalMemoryBytes != null && totalMemoryBytes < 0)
            throw new IllegalArgumentException("totalMemoryBytes must be non-negative");
        if (freeMemoryBytes != null && freeMemoryBytes < 0)
            throw new IllegalArgumentException("freeMemoryBytes must be non-negative");
        if (totalMemoryBytes != null && freeMemoryBytes != null && freeMemoryBytes > totalMemoryBytes)
            throw new IllegalArgumentException("freeMemoryBytes must not exceed totalMemoryBytes");
        vendor = normalize(vendor);
        architecture = normalize(architecture);
        driverVersion = normalize(driverVersion);
        runtimeVersion = normalize(runtimeVersion);
    }

    private static String requireText(String value, String field) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(field + " must not be blank");
        return value.trim();
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}