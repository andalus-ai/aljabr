package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.MetadataValue;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorsTargetCapabilityTest {

    @Test
    void preservesTypedMetadataThroughSafetensorsMetadataEnvelope() {
        var model = new ModelDescriptor(
                ModelIdentity.of("typed-metadata"),
                "llama",
                "gguf",
                Map.of("general.architecture", "llama"),
                Map.of(),
                List.of()).withTypedMetadata(Map.of("general.alignment",
                        new MetadataValue.Scalar(MetadataValue.Kind.UINT32, 32L)));

        CompatibilityReport report = new SafetensorsTargetCapability().assess(model);

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
        assertTrue(report.diagnostics().stream().noneMatch(d ->
                d.code().equals("GGUF_TYPED_METADATA_FLATTENED")));
    }
    @Test
    void treatsModelAndTensorStringExtensionsAsPreservedByEnvelope() {
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L),
                "contiguous", null, null, Map.of("vendor.attribute", "value"));
        var model = new ModelDescriptor(
                ModelIdentity.of("opaque-extensions"),
                "llama",
                "safetensors",
                Map.of("general.architecture", "llama"),
                Map.of("vendor.model.attribute", "model-value"),
                List.of(tensor));

        CompatibilityReport report = new SafetensorsTargetCapability().assess(model);

        assertEquals(CompatibilityReport.ConversionFidelity.FULL, report.fidelity());
        assertTrue(report.diagnostics().stream().noneMatch(d ->
                d.code().equals("OPAQUE_METADATA_REQUIRES_MAPPING")));
    }

    @Test
    void marksLegacyFlattenedGgufMetadataAsPartial() {
        var model = new ModelDescriptor(
                ModelIdentity.of("legacy-typed-metadata"),
                "llama",
                "gguf",
                Map.of("general.architecture", "llama"),
                Map.of("gguf.metadata.general.alignment", "32"),
                List.of());

        CompatibilityReport report = new SafetensorsTargetCapability().assess(model);

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, report.fidelity());
        assertTrue(report.diagnostics().stream().anyMatch(d ->
                d.code().equals("GGUF_TYPED_METADATA_FLATTENED")));
    }

}
