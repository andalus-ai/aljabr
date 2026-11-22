package tech.kayys.aljabr.diagnostics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.time.Instant;

import org.junit.jupiter.api.Test;

class DiagnosticContextIntegrationTest {

    @Test
    void attachesTypedContextWithoutChangingLegacyDiagnosticShape() {
        DiagnosticContext context = DiagnosticContext.builder()
                .correlationId("trace-42")
                .parentId("scope-7")
                .operation("matmul")
                .component("tensor")
                .provider("cpu")
                .attribute("dtype", "F32")
                .build();

        Diagnostic diagnostic = Diagnostic.withContext(
                "tensor.operation.failed",
                DiagnosticSeverity.ERROR,
                "tensor",
                "Matrix multiplication failed",
                context,
                Instant.EPOCH);

        assertEquals(Instant.EPOCH, diagnostic.occurredAt());
        assertEquals("trace-42", diagnostic.context().get("correlation.id"));
        assertEquals("scope-7", diagnostic.context().get("parent.id"));
        assertEquals("matmul", diagnostic.context().get("operation"));
        assertEquals("cpu", diagnostic.context().get("provider"));
        assertEquals("F32", diagnostic.context().get("dtype"));
    }

    @Test
    void exposesOneSharedNoopSink() {
        assertSame(DiagnosticSink.NOOP, DiagnosticSink.noop());
        assertSame(DiagnosticSink.noop(), DiagnosticSink.noop());
    }
}
