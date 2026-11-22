package tech.kayys.aljabr.diagnostics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DiagnosticContextTest {

    @Test
    void createsStableImmutableContextMap() {
        DiagnosticContext context = DiagnosticContext.builder()
                .correlationId("trace-1")
                .parentId("operation-0")
                .operation("matmul")
                .component("tensor")
                .provider("cpu")
                .attribute("dtype", "F32")
                .attribute("shape", "[2, 3]")
                .build();

        assertEquals("trace-1", context.correlationId());
        assertEquals("operation-0", context.parentId());
        assertEquals("matmul", context.asMap().get("operation"));
        assertEquals("cpu", context.asMap().get("provider"));
        assertEquals("F32", context.asMap().get("dtype"));
        assertThrows(UnsupportedOperationException.class,
                () -> context.asMap().put("unsafe", "value"));
    }

    @Test
    void copiesCallerAttributes() {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("dtype", "F32");

        DiagnosticContext context = new DiagnosticContext(
                null, null, null, null, null, attributes);
        attributes.put("dtype", "F64");

        assertEquals("F32", context.attributes().get("dtype"));
    }

    @Test
    void rejectsUnboundedOrMalformedAttributes() {
        assertThrows(IllegalArgumentException.class,
                () -> DiagnosticContext.builder().attribute("bad key", "value"));
        assertThrows(IllegalArgumentException.class,
                () -> DiagnosticContext.builder().attribute("correlation.id", "spoofed"));
        assertThrows(IllegalArgumentException.class,
                () -> DiagnosticContext.builder().attribute("payload", "x".repeat(513)));
        assertThrows(IllegalArgumentException.class,
                () -> DiagnosticContext.builder().correlationId("x".repeat(129)));

        DiagnosticContext.Builder builder = DiagnosticContext.builder();
        for (int i = 0; i < DiagnosticContext.MAX_ATTRIBUTES; i++) {
            builder.attribute("key" + i, "value");
        }
        assertThrows(IllegalArgumentException.class, () -> builder.attribute("overflow", "value"));
    }

    @Test
    void rejectsConflictsWithStandardKeys() {
        DiagnosticContext context = DiagnosticContext.builder()
                .correlationId("trace-1")
                .attribute("operation", "overridden")
                .build();

        assertThrows(IllegalArgumentException.class, context::asMap);
    }
}
