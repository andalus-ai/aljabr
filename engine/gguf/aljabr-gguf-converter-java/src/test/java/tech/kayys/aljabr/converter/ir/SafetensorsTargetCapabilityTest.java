package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorsTargetCapabilityTest {

    @Test
    void marksFlattenedTypedGgufMetadataAsPartialFidelity() {
        var model = new ModelDescriptor(
                ModelIdentity.of("typed-metadata"),
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
