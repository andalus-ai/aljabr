package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Metadata-only reference to an external model asset.
 *
 * <p>The location is an opaque URI or resolver-specific locator. AMIR never
 * opens or owns the referenced payload as a side effect of constructing this
 * descriptor.
 */
public record AssetReference(
        String id,
        String location,
        Optional<String> mediaType,
        OptionalLong sizeBytes,
        Optional<String> checksum,
        Optional<String> checksumAlgorithm,
        Map<String, String> extensions) {

    public AssetReference {
        id = requireText(id, "id");
        location = requireText(location, "location");
        mediaType = Objects.requireNonNull(mediaType, "mediaType must not be null");
        sizeBytes = Objects.requireNonNull(sizeBytes, "sizeBytes must not be null");
        checksum = Objects.requireNonNull(checksum, "checksum must not be null");
        checksumAlgorithm = Objects.requireNonNull(checksumAlgorithm, "checksumAlgorithm must not be null");
        extensions = immutableMap(extensions);
        mediaType.ifPresent(value -> requireText(value, "mediaType"));
        if (sizeBytes.isPresent() && sizeBytes.getAsLong() < 0) {
            throw new IllegalArgumentException("sizeBytes must not be negative");
        }
        checksum.ifPresent(value -> requireText(value, "checksum"));
        checksumAlgorithm.ifPresent(value -> requireText(value, "checksumAlgorithm"));
        if (checksum.isPresent() != checksumAlgorithm.isPresent()) {
            throw new IllegalArgumentException("checksum and checksumAlgorithm must be provided together");
        }
    }

    public static AssetReference of(String id, String location) {
        return new AssetReference(id, location, Optional.empty(), OptionalLong.empty(),
                Optional.empty(), Optional.empty(), Map.of());
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }

    private static Map<String, String> immutableMap(Map<String, String> values) {
        Objects.requireNonNull(values, "extensions must not be null");
        if (values.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null)) {
            throw new IllegalArgumentException("extensions must contain non-blank keys and non-null values");
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(values));
    }
}
