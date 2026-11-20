package tech.kayys.aljabr.diagnostics;

import java.io.PrintStream;
import java.util.Objects;

/**
 * Receives diagnostics. Implementations are invoked synchronously by the
 * caller; exceptions from a sink are deliberately not swallowed.
 */
@FunctionalInterface
public interface DiagnosticSink {

    /** Receives one diagnostic. */
    void accept(Diagnostic diagnostic);

    /** Returns a sink that intentionally discards all diagnostics. */
    static DiagnosticSink noop() {
        return diagnostic -> {
            // Intentionally disabled.
        };
    }

    /**
     * Returns a simple text sink writing one formatted diagnostic per line.
     * The supplied stream remains owned by the caller and is never closed.
     */
    static DiagnosticSink console(PrintStream output) {
        Objects.requireNonNull(output, "output");
        return diagnostic -> output.println(DiagnosticFormatter.format(diagnostic));
    }
}
