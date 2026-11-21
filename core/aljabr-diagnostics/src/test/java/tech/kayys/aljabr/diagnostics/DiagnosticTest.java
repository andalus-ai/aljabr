package tech.kayys.aljabr.diagnostics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DiagnosticTest {

    @Test
    void copiesAndOrdersContext() {
        Map<String, String> source = new HashMap<>();
        source.put("provider", "cpu");
        source.put("operation", "matmul");

        Diagnostic diagnostic = new Diagnostic(
                "tensor.operation.fallback",
                DiagnosticSeverity.WARNING,
                "execution",
                "Falling back to CPU",
                source,
                Instant.EPOCH);

        source.put("later", "mutation");

        assertEquals(Map.of("operation", "matmul", "provider", "cpu"), diagnostic.context());
        assertEquals("operation", diagnostic.context().keySet().iterator().next());
        assertThrows(UnsupportedOperationException.class,
                () -> diagnostic.context().put("unsafe", "value"));
    }

    @Test
    void rejectsMissingOrBlankRequiredFields() {
        assertThrows(NullPointerException.class,
                () -> new Diagnostic(null, DiagnosticSeverity.ERROR, "tensor", "Invalid shape"));
        assertThrows(IllegalArgumentException.class,
                () -> new Diagnostic(" ", DiagnosticSeverity.ERROR, "tensor", "Invalid shape"));
        assertThrows(NullPointerException.class,
                () -> new Diagnostic("tensor.invalid", null, "tensor", "Invalid shape"));
        assertThrows(IllegalArgumentException.class,
                () -> new Diagnostic("tensor.invalid", DiagnosticSeverity.ERROR, " ", "Invalid shape"));
    }
}
