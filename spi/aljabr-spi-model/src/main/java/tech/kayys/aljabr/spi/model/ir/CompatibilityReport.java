package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

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

    /**
     * One preflight finding. A path identifies the exact descriptor field or
     * metadata key affected; it is empty for global findings.
     */
    public record Diagnostic(Severity severity, String code, String message, Optional<String> path) {
        /** Compatibility constructor for diagnostics without a structured path. */
        public Diagnostic(Severity severity, String code, String message) {
            this(severity, code, message, Optional.empty());
        }

        /** Creates a diagnostic associated with an exact AMIR descriptor path. */
        public Diagnostic(Severity severity, String code, String message, String path) {
            this(severity, code, message, Optional.of(requirePath(path)));
        }

        public Diagnostic {
            Objects.requireNonNull(severity, "severity must not be null");
            if (code == null || code.isBlank()) {
                throw new IllegalArgumentException("code must not be blank");
            }
            if (message == null || message.isBlank()) {
                throw new IllegalArgumentException("message must not be blank");
            }
            path = Objects.requireNonNull(path, "path must not be null");
            path.ifPresent(CompatibilityReport.Diagnostic::requirePath);
        }

        private static String requirePath(String path) {
            if (path == null || path.isBlank()) {
                throw new IllegalArgumentException("path must not be blank");
            }
            return path;
        }
    }
}
