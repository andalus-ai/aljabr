package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Typed, format-neutral description of a model architecture.
 *
 * <p>The architecture identifier is intentionally independent of the family
 * classification. Unknown architecture identifiers and capabilities remain
 * representable without requiring AMIR to understand their semantics.
 */
public record ArchitectureDescriptor(
        String id,
        ArchitectureFamily family,
        ArchitectureConfiguration configuration,
        Set<String> capabilities,
        Map<String, String> extensions) {

    public ArchitectureDescriptor {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
        Objects.requireNonNull(family, "family must not be null");
        configuration = Objects.requireNonNull(configuration, "configuration must not be null");
        capabilities = immutableIdentifiers(capabilities, "capabilities");
        extensions = immutableExtensions(extensions);
    }

    public static ArchitectureDescriptor of(
            String id,
            ArchitectureFamily family,
            ArchitectureConfiguration configuration) {
        return new ArchitectureDescriptor(id, family, configuration, Set.of(), Map.of());
    }

    private static Set<String> immutableIdentifiers(Set<String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        var copy = new TreeSet<String>();
        for (String value : values) {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(name + " must contain non-blank identifiers");
            }
            copy.add(value);
        }
        return Collections.unmodifiableSet(copy);
    }

    private static Map<String, String> immutableExtensions(Map<String, String> values) {
        Objects.requireNonNull(values, "extensions must not be null");
        var copy = new LinkedHashMap<String, String>();
        values.forEach((key, value) -> {
            if (key == null || key.isBlank() || value == null) {
                throw new IllegalArgumentException("extensions must contain non-blank keys and non-null values");
            }
            copy.put(key, value);
        });
        return Collections.unmodifiableMap(copy);
    }
}
