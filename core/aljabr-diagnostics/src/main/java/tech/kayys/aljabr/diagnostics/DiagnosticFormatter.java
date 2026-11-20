package tech.kayys.aljabr.diagnostics;

import java.util.Map;
import java.util.Objects;

/** Deterministic, single-line text rendering for a structured diagnostic. */
public final class DiagnosticFormatter {

    private DiagnosticFormatter() {
    }

    /**
     * Formats fields in a stable order. Context keys are already sorted by
     * {@link Diagnostic}; control characters are escaped to keep one event on
     * one line.
     */
    public static String format(Diagnostic diagnostic) {
        Objects.requireNonNull(diagnostic, "diagnostic");
        StringBuilder result = new StringBuilder()
                .append(diagnostic.occurredAt())
                .append(' ')
                .append(diagnostic.severity())
                .append(' ')
                .append(diagnostic.code())
                .append(" [")
                .append(diagnostic.category())
                .append("]: ")
                .append(escape(diagnostic.message()));

        if (!diagnostic.context().isEmpty()) {
            result.append(" {");
            boolean first = true;
            for (Map.Entry<String, String> entry : diagnostic.context().entrySet()) {
                if (!first) {
                    result.append(", ");
                }
                result.append(escape(entry.getKey()))
                        .append('=')
                        .append(escape(entry.getValue()));
                first = false;
            }
            result.append('}');
        }
        return result.toString();
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
                .replace("\t", "\\t");
    }
}
