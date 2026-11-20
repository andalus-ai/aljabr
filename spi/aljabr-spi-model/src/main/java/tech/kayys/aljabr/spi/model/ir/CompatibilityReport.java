package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;

/**
 * Explicit statement of how faithfully a model can be represented by a target.
 * A conversion must not silently treat partial compatibility as lossless.
 */
public record CompatibilityReport(
        ConversionFidelity fidelity,
        List<Diagnostic> diagnostics) {

    public CompatibilityReport {
        Objects.requireNonNull(fidelity, "fidelity must not be null");
        diagnostics = List.copyOf(Objects.requireNonNull(diagnostics, "diagnostics must not be null"));
        diagnostics.forEach(diagnostic -> Objects.requireNonNull(diagnostic, "diagnostic must not be null"));
    }

    public enum ConversionFidelity {
        /** All required model semantics and metadata can be preserved. */
        FULL,
        /** Conversion is possible, but one or more semantics/metadata need explicit handling. */
        PARTIAL,
        /** The target cannot faithfully represent the required model semantics. */
        UNSUPPORTED
    }

    public enum Severity {
        INFO,
        WARNING,
        ERROR
    }

    public record Diagnostic(Severity severity, String code, String message) {
        public Diagnostic {
            Objects.requireNonNull(severity, "severity must not be null");
            if (code == null || code.isBlank()) {
                throw new IllegalArgumentException("code must not be blank");
            }
            if (message == null || message.isBlank()) {
                throw new IllegalArgumentException("message must not be blank");
            }
        }
    }
}
