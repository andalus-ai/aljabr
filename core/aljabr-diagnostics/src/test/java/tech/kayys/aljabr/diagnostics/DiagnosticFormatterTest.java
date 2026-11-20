package tech.kayys.aljabr.diagnostics;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.Instant;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DiagnosticFormatterTest {

    @Test
    void formatsDeterministicallyAndEscapesLineBreaks() {
        Diagnostic diagnostic = new Diagnostic(
                "tensor.shape.mismatch",
                DiagnosticSeverity.ERROR,
                "tensor",
                "Expected [2, 3]\nbut received [3, 2]",
                Map.of("shape", "[3, 2]", "operation", "matmul"),
                Instant.parse("2026-01-02T03:04:05Z"));

        assertEquals(
                "2026-01-02T03:04:05Z ERROR tensor.shape.mismatch [tensor]: "
                        + "Expected [2, 3]\\nbut received [3, 2] "
                        .trim()
                        + "{operation=matmul, shape=[3, 2]}",
                DiagnosticFormatter.format(diagnostic));
    }
}
