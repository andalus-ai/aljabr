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
                tensor("weight", "IQ4_NL", List.of(15L, 15L))));

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
    void safetensorsClassifiesSupportedGgmlBlockTypesAsPartialDequantization() {
        var report = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("weight", "Q4_K", List.of(256L))));

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, report.fidelity());
        assertEquals("GGML_QUANTIZATION_DEQUANTIZED", report.diagnostics().getFirst().code());
    }

    @Test
    void safetensorsAcceptsNativeScalarTypes() {
        var report = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("weight", "BF16", List.of(4L, 8L))));

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
    }

    @Test
    void q4AndQ5VariantsAndQ8ZeroAreSupportedWithExplicitLossyDequantization() {
        var q4 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q4", "Q4_0", List.of(32L))));
        var q4_1 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q4_1", "Q4_1", List.of(32L))));
        var q5_0 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q5_0", "Q5_0", List.of(32L))));
        var q5_1 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q5_1", "Q5_1", List.of(32L))));
        var q6 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q6", "Q6_K", List.of(256L))));
        var q2 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q2", "Q2_K", List.of(256L))));
        var q3 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q3", "Q3_K", List.of(256L))));
        var q4k = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q4k", "Q4_K", List.of(256L))));
        var q5k = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q5k", "Q5_K", List.of(256L))));
        var q8 = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("q8", "Q8_0", List.of(32L))));
        var iq3xxs = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("iq3xxs", "IQ3_XXS", List.of(256L))));
        var iq3s = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("iq3s", "IQ3_S", List.of(256L))));
        var iq4nl = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("iq4nl", "IQ4_NL", List.of(32L))));
        var iq4xs = new SafetensorsTargetCapability().assess(model("llama", "gguf",
                tensor("iq4xs", "IQ4_XS", List.of(256L))));

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q4.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q4_1.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q5_0.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q5_1.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q6.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q2.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q3.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q4k.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q5k.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, q8.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, iq3xxs.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, iq3s.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, iq4nl.fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, iq4xs.fidelity());
        assertEquals("GGML_QUANTIZATION_DEQUANTIZED", q4.diagnostics().getFirst().code());
    }

    @Test
    void opaqueExtensionsArePreservedInSafetensorsFileMetadata() {
        var source = new ModelDescriptor(
                ModelIdentity.of("test"), "llama", "gguf", Map.of(),
                Map.of("gguf.metadata.json", "{\"custom\":true}"),
                List.of(tensor("weight", "F32", List.of(2L))));

        var report = new SafetensorsTargetCapability().assess(source);

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
    }

    @Test
    void nonGgufOpaqueExtensionsRemainPartialWhenNotMappedByExecutor() {
        var source = new ModelDescriptor(ModelIdentity.of("test"), "llama", "safetensors",
                Map.of(), Map.of("vendor.extra", "value"), List.of(tensor("weight", "F32", List.of(2L))));

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL,
                new SafetensorsTargetCapability().assess(source).fidelity());
    }

    private static ModelDescriptor model(String architecture, String format, TensorDescriptor tensor) {
        return new ModelDescriptor(ModelIdentity.of("test"), architecture, format,
                Map.of(), Map.of(), List.of(tensor));
    }

    private static TensorDescriptor tensor(String name, String dtype, List<Long> shape) {
        return new TensorDescriptor(name, dtype, shape, "unknown", null, null);
    }
}
