package tech.kayys.aljabr.spi.model.ir;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable typed architecture configuration.
 *
 * <p>Standard properties are represented as typed values. Source-specific or
 * not-yet-standardized values are retained in the opaque extension map.
 */
public record ArchitectureConfiguration(
        Map<String, ConfigurationValue> properties,
        Map<String, String> extensions) {

    public ArchitectureConfiguration {
        properties = immutableProperties(properties);
        extensions = immutableStringMap(extensions);
    }

    public static ArchitectureConfiguration empty() {
        return new ArchitectureConfiguration(Map.of(), Map.of());
    }

    public Optional<ConfigurationValue> find(String name) {
        return Optional.ofNullable(properties.get(requireName(name)));
    }

    public Optional<String> stringValue(String name) {
        return find(name).filter(ConfigurationValue.StringValue.class::isInstance)
                .map(ConfigurationValue.StringValue.class::cast)
                .map(ConfigurationValue.StringValue::value);
    }

    public Optional<Long> integerValue(String name) {
        return find(name).filter(ConfigurationValue.IntegerValue.class::isInstance)
                .map(ConfigurationValue.IntegerValue.class::cast)
                .map(ConfigurationValue.IntegerValue::value);
    }

    public Optional<Boolean> booleanValue(String name) {
        return find(name).filter(ConfigurationValue.BooleanValue.class::isInstance)
                .map(ConfigurationValue.BooleanValue.class::cast)
                .map(ConfigurationValue.BooleanValue::value);
    }

    private static Map<String, ConfigurationValue> immutableProperties(
            Map<String, ConfigurationValue> values) {
        Objects.requireNonNull(values, "properties must not be null");
        var copy = new LinkedHashMap<String, ConfigurationValue>();
        values.forEach((key, value) -> {
            requireName(key);
            copy.put(key, Objects.requireNonNull(value, "configuration value must not be null"));
        });
        return Collections.unmodifiableMap(copy);
    }

    private static Map<String, String> immutableStringMap(Map<String, String> values) {
        Objects.requireNonNull(values, "extensions must not be null");
        var copy = new LinkedHashMap<String, String>();
        values.forEach((key, value) -> {
            requireName(key);
            copy.put(key, Objects.requireNonNull(value, "extension value must not be null"));
        });
        return Collections.unmodifiableMap(copy);
    }

    private static String requireName(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("configuration property name must not be blank");
        }
        return name;
    }
}
