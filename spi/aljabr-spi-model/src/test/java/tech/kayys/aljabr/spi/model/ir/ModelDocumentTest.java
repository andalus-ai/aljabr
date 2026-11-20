package tech.kayys.aljabr.spi.model.ir;

import org.junit.jupiter.api.Test;

import java.net.URI;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelDocumentTest {

    @Test
    void retainsUnknownExtensionsAndLazyPayloadReferences() {
        var payload = new PayloadReference(
                URI.create("file:///models/checkpoint.safetensors"),
                "model-00001-of-00002.safetensors",
                128L,
                4096L,
                PayloadReference.OffsetBasis.DATA_SECTION);
        var tensor = new TensorDescriptor(
                "model.layers.0.self_attn.q_proj.weight",
                "F16",
                List.of(4096L, 4096L),
                "row-major",
                null,
                payload,
                Map.of("vendor.quant_hint", "{\"scheme\":\"unknown\"}"));

        var document = new ModelDocument(
                "checkpoint-1",
                "unknown-architecture",
                "safetensors",
                Map.of("format_version", "1"),
                Map.of("vendor.unknown_config", "{\"preserve\":true}"),
                List.of(tensor));

        assertEquals("unknown-architecture", document.architecture());
        assertEquals("{\"preserve\":true}", document.extensions().get("vendor.unknown_config"));
        assertEquals(PayloadReference.OffsetBasis.DATA_SECTION, document.tensors().get(0).payload().offsetBasis());
        assertEquals(4096L, document.tensors().get(0).payload().length());
        assertThrows(UnsupportedOperationException.class,
                () -> document.extensions().put("lost", "no"));
    }

    @Test
    void rejectsDuplicateTensorNames() {
        var first = new TensorDescriptor("weight", "F32", List.of(2L), "row-major", null, null);
        var second = new TensorDescriptor("weight", "F16", List.of(2L), "row-major", null, null);

        assertThrows(IllegalArgumentException.class, () -> new ModelDocument(
                "model", "unknown", "safetensors", Map.of(), Map.of(), List.of(first, second)));
    }

    @Test
    void rejectsPayloadRangeOverflow() {
        assertThrows(IllegalArgumentException.class, () -> new PayloadReference(
                URI.create("file:///model.bin"), "model.bin", Long.MAX_VALUE, 1L,
                PayloadReference.OffsetBasis.CONTAINER));
    }

    @Test
    void reportsPartialCompatibilityWithoutHidingDiagnostics() {
        var report = new CompatibilityReport(
                CompatibilityReport.ConversionFidelity.PARTIAL,
                List.of(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.WARNING,
                        "UNKNOWN_OPERATOR",
                        "Target format has no equivalent for the source operator.")));

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, report.fidelity());
        assertEquals("UNKNOWN_OPERATOR", report.diagnostics().get(0).code());
    }
}
