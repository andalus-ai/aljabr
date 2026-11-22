package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.MetadataValue;

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
}
