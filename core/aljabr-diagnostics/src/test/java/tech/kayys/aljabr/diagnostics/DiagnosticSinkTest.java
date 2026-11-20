package tech.kayys.aljabr.diagnostics;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DiagnosticSinkTest {

    @Test
    void noopSinkAcceptsDiagnostic() {
        DiagnosticSink.noop().accept(
                new Diagnostic("runtime.notice", DiagnosticSeverity.INFO, "runtime", "No-op"));
    }

    @Test
    void consoleSinkWritesOneFormattedLine() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        PrintStream stream = new PrintStream(bytes, true, StandardCharsets.UTF_8);
        Diagnostic diagnostic = new Diagnostic(
                "provider.unavailable",
                DiagnosticSeverity.WARNING,
                "provider",
                "Provider unavailable",
                Map.of("provider", "example"),
                Instant.EPOCH);

        DiagnosticSink.console(stream).accept(diagnostic);

        assertEquals(
                "1970-01-01T00:00:00Z WARNING provider.unavailable [provider]: "
                        + "Provider unavailable {provider=example}"
                        + System.lineSeparator(),
                bytes.toString(StandardCharsets.UTF_8));
    }
}
