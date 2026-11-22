package tech.kayys.aljabr.core.tool;

import java.util.Objects;

/**
 * Machine-readable diagnostic emitted while extracting tool calls from
 * untrusted model output.
 *
 * @param code stable, adapter-defined diagnostic code
 * @param severity diagnostic severity
 * @param message human-readable explanation
 * @param offset zero-based character offset in the source, or {@code -1}
 *               when the adapter cannot provide one
 */
public record ToolCallParseDiagnostic(
        String code,
        Severity severity,
        String message,
        int offset) {

    public ToolCallParseDiagnostic {
        Objects.requireNonNull(code, "code");
        Objects.requireNonNull(severity, "severity");
        Objects.requireNonNull(message, "message");
        if (code.isBlank()) throw new IllegalArgumentException("code must not be blank");
        if (message.isBlank()) throw new IllegalArgumentException("message must not be blank");
        if (offset < -1) throw new IllegalArgumentException("offset must be -1 or non-negative");
    }

    public enum Severity {
        WARNING,
        ERROR
    }
}
