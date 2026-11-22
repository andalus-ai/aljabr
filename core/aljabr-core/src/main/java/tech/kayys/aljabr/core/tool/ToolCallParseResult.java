package tech.kayys.aljabr.core.tool;

import java.util.List;
import java.util.Objects;

/**
 * Immutable outcome of parsing untrusted model output into tool calls.
 *
 * <p>A result may contain both valid calls and diagnostics when an adapter
 * can recover from malformed segments. An empty result with no diagnostics
 * means that no tool calls were present. This type does not authorize,
 * resolve, dispatch, or execute calls.</p>
 */
public record ToolCallParseResult(
        List<ToolCall> calls,
        List<ToolCallParseDiagnostic> diagnostics) {

    public ToolCallParseResult {
        calls = List.copyOf(Objects.requireNonNull(calls, "calls"));
        diagnostics = List.copyOf(Objects.requireNonNull(diagnostics, "diagnostics"));
    }

    /** Returns whether at least one complete call was extracted. */
    public boolean hasCalls() {
        return !calls.isEmpty();
    }

    /** Returns whether any diagnostic represents an error. */
    public boolean hasErrors() {
        return diagnostics.stream()
                .anyMatch(diagnostic -> diagnostic.severity() == ToolCallParseDiagnostic.Severity.ERROR);
    }

    /** Returns whether parsing completed without emitting diagnostics. */
    public boolean isComplete() {
        return diagnostics.isEmpty();
    }

    /** Returns whether no calls and no diagnostics were produced. */
    public boolean isEmpty() {
        return calls.isEmpty() && diagnostics.isEmpty();
    }
}
