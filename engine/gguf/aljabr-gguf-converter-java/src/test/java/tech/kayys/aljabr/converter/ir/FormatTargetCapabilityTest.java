package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FormatTargetCapabilityTest {

    @Test
    void ggufAcceptsRepresentableFloatingPointTensor() {
        var report = new GgufTargetCapability().assess(model("llama", "safetensors",
                tensor("weight", "F16", List.of(2L, 4L))));

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
    }

    @Test
    void ggufRejectsQuantizedTensorWithInvalidBlockElementCount() {
        var report = new GgufTargetCapability().assess(model("llama", "safetensors",
                tensor("weight", "Q4_K", List.of(15L, 15L))));

        assertEquals(CompatibilityReport.ConversionFidelity.UNSUPPORTED, report.fidelity());
        assertEquals("GGUF_QUANTIZATION_BLOCK_MISMATCH", report.diagnostics().getFirst().code());
    }

    @Test
    void ggufRejectsDtypeWithoutGgmlStorageType() {
        var report = new GgufTargetCapability().assess(model("llama", "safetensors",
                tensor("weight", "F8_E4M3", List.of(32L))));

        assertEquals(CompatibilityReport.ConversionFidelity.UNSUPPORTED, report.fidelity());
        assertEquals("GGUF_DTYPE_UNSUPPORTED", report.diagnostics().getFirst().code());
    }

    @Test
    void ggufPreservesOpaqueStringExtensionsAsNamespacedMetadata() {
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L), "unknown", null,
                null, Map.of("vendor.layout", "row-major"));
        var source = new ModelDescriptor(ModelIdentity.of("test"), "llama", "safetensors",
                Map.of("model_type", "llama"), Map.of("vendor.source", "checkpoint"),
                List.of(tensor));

        var report = new GgufTargetCapability().assess(source);

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
    }

    @Test
    void safetensorsRequiresExplicitDequantizationForGgmlBlockTypes() {
        var report = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("weight", "Q4_K", List.of(256L))));

        assertEquals(CompatibilityReport.ConversionFidelity.UNSUPPORTED, report.fidelity());
        assertEquals("SAFETENSORS_DTYPE_UNSUPPORTED", report.diagnostics().getFirst().code());
    }

    @Test
    void safetensorsAcceptsNativeScalarTypes() {
        var report = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("weight", "BF16", List.of(4L, 8L))));

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
    }

    @Test
    void opaqueExtensionsMakeSafetensorsAssessmentPartial() {
        var source = new ModelDescriptor(
                ModelIdentity.of("test"), "llama", "gguf", Map.of(),
                Map.of("gguf.metadata.json", "{\"custom\":true}"),
                List.of(tensor("weight", "F32", List.of(2L))));

        var report = new SafetensorsTargetCapability().assess(source);

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, report.fidelity());
    }

    private static ModelDescriptor model(String architecture, String format, TensorDescriptor tensor) {
        return new ModelDescriptor(ModelIdentity.of("test"), architecture, format,
                Map.of(), Map.of(), List.of(tensor));
    }

    private static TensorDescriptor tensor(String name, String dtype, List<Long> shape) {
        return new TensorDescriptor(name, dtype, shape, "unknown", null, null);
    }
}
