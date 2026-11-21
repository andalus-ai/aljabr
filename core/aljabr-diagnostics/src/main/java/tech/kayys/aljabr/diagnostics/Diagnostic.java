package tech.kayys.aljabr.diagnostics;

import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Immutable, structured description of a noteworthy condition in Aljabr.
 *
 * <p>Codes and categories are strings so independently published modules can
 * add stable identifiers without modifying a central enum. Context is limited
 * to strings: callers must explicitly choose which metadata is safe to expose.
 * Tensor values, prompts, model weights and other payloads are never captured
 * automatically.</p>
 *
 * @param code stable diagnostic identifier, for example {@code tensor.shape.mismatch}
 * @param severity relative importance of the condition
 * @param category stable domain/category identifier
 * @param message concise human-readable explanation
 * @param context explicitly supplied, non-sensitive string metadata
 * @param occurredAt time at which the condition was observed
 */
public record Diagnostic(
        String code,
        DiagnosticSeverity severity,
        String category,
        String message,
        Map<String, String> context,
        Instant occurredAt) {

    public Diagnostic {
        code = requireText(code, "code");
        severity = Objects.requireNonNull(severity, "severity");
        category = requireText(category, "category");
        message = requireText(message, "message");
        occurredAt = Objects.requireNonNull(occurredAt, "occurredAt");

        var copy = new TreeMap<String, String>();
        Objects.requireNonNull(context, "context").forEach((key, value) ->
                copy.put(requireText(key, "context key"), Objects.requireNonNull(value, "context value")));
        context = Collections.unmodifiableMap(copy);
    }

    /** Creates a diagnostic timestamped at construction time. */
    public Diagnostic(
            String code,
            DiagnosticSeverity severity,
            String category,
            String message,
            Map<String, String> context) {
        this(code, severity, category, message, context, Instant.now());
    }

    /** Creates a diagnostic without additional context. */
    public Diagnostic(String code, DiagnosticSeverity severity, String category, String message) {
        this(code, severity, category, message, Map.of(), Instant.now());
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return normalized;
    }
}
