package tech.kayys.aljabr.diagnostics;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Explicit, immutable execution context attached to a diagnostic.
 *
 * <p>Identifiers are opaque correlation values, not places to store user data.
 * Standard fields are kept separate from bounded, caller-allowlisted
 * attributes. This type performs structural validation; callers and exporters
 * remain responsible for deciding whether an attribute is safe to disclose.</p>
 */
public record DiagnosticContext(
        String correlationId,
        String parentId,
        String operation,
        String component,
        String provider,
        Map<String, String> attributes) {

    public static final int MAX_ATTRIBUTES = 32;
    public static final int MAX_KEY_LENGTH = 64;
    public static final int MAX_VALUE_LENGTH = 512;

    private static final DiagnosticContext EMPTY =
            new DiagnosticContext(null, null, null, null, null, Map.of());

    public DiagnosticContext {
        correlationId = optionalText(correlationId, "correlationId", 128);
        parentId = optionalText(parentId, "parentId", 128);
        operation = optionalText(operation, "operation", 128);
        component = optionalText(component, "component", 128);
        provider = optionalText(provider, "provider", 128);

        Objects.requireNonNull(attributes, "attributes");
        if (attributes.size() > MAX_ATTRIBUTES) {
            throw new IllegalArgumentException("attributes must contain at most " + MAX_ATTRIBUTES + " entries");
        }

        var copy = new TreeMap<String, String>();
        attributes.forEach((key, value) -> {
            String safeKey = requireAttributeKey(key);
            String safeValue = Objects.requireNonNull(value, "attribute value");
            if (safeValue.length() > MAX_VALUE_LENGTH) {
                throw new IllegalArgumentException("attribute value exceeds " + MAX_VALUE_LENGTH + " characters");
            }
            copy.put(safeKey, safeValue);
        });
        attributes = Collections.unmodifiableMap(copy);
    }

    /** Returns an empty context without allocating a new instance. */
    public static DiagnosticContext empty() {
        return EMPTY;
    }

    /** Starts building an explicit diagnostic context. */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Converts populated standard fields and attributes to a stable,
     * immutable string map suitable for the existing Diagnostic API.
     */
    public Map<String, String> asMap() {
        if (correlationId == null && parentId == null && operation == null
                && component == null && provider == null && attributes.isEmpty()) {
            return Map.of();
        }
        var result = new TreeMap<String, String>();
        putIfPresent(result, "correlation.id", correlationId);
        putIfPresent(result, "parent.id", parentId);
        putIfPresent(result, "operation", operation);
        putIfPresent(result, "component", component);
        putIfPresent(result, "provider", provider);
        attributes.forEach((key, value) -> {
            if (result.putIfAbsent(key, value) != null) {
                throw new IllegalArgumentException("attribute conflicts with standard context key: " + key);
            }
        });
        return Collections.unmodifiableMap(result);
    }

    private static void putIfPresent(Map<String, String> target, String key, String value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private static String optionalText(String value, String name, int maxLength) {
        if (value == null) {
            return null;
        }
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank when supplied");
        }
        if (normalized.length() > maxLength) {
            throw new IllegalArgumentException(name + " exceeds " + maxLength + " characters");
        }
        return normalized;
    }

    private static String requireAttributeKey(String key) {
        Objects.requireNonNull(key, "attribute key");
        String normalized = key.trim();
        if (normalized.isEmpty() || normalized.length() > MAX_KEY_LENGTH
                || !normalized.matches("[a-zA-Z][a-zA-Z0-9_.-]*")) {
            throw new IllegalArgumentException("invalid diagnostic attribute key: " + key);
        }
        return normalized;
    }

    /** Mutable builder; the resulting context is immutable. */
    public static final class Builder {
        private String correlationId;
        private String parentId;
        private String operation;
        private String component;
        private String provider;
        private final Map<String, String> attributes = new TreeMap<>();

        public Builder correlationId(String value) {
            correlationId = value;
            return this;
        }

        public Builder parentId(String value) {
            parentId = value;
            return this;
        }

        public Builder operation(String value) {
            operation = value;
            return this;
        }

        public Builder component(String value) {
            component = value;
            return this;
        }

        public Builder provider(String value) {
            provider = value;
            return this;
        }

        /** Adds one explicitly selected metadata attribute. */
        public Builder attribute(String key, String value) {
            String safeKey = requireAttributeKey(key);
            Objects.requireNonNull(value, "attribute value");
            if (value.length() > MAX_VALUE_LENGTH) {
                throw new IllegalArgumentException("attribute value exceeds " + MAX_VALUE_LENGTH + " characters");
            }
            if (!attributes.containsKey(safeKey) && attributes.size() >= MAX_ATTRIBUTES) {
                throw new IllegalArgumentException("attributes must contain at most " + MAX_ATTRIBUTES + " entries");
            }
            attributes.put(safeKey, value);
            return this;
        }

        public DiagnosticContext build() {
            return new DiagnosticContext(correlationId, parentId, operation, component, provider, attributes);
        }
    }
}
