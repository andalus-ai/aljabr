package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelConversionPlanners;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BuiltInCapabilityDiscoveryTest {

    @Test
    void discoversBuiltInTargetFormatCapabilities() {
        var planner = ModelConversionPlanners.discover();

        assertTrue(planner.supportsTarget("gguf"));
        assertTrue(planner.supportsTarget("safetensors"));
    }

    @Test
    void centralPlannerUsesPathAwareSharedFidelityAssessment() {
        var planner = ModelConversionPlanners.discover();
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L),
                "contiguous", null, null, Map.of("vendor.scale", "0.5"));
        var model = new ModelDescriptor(ModelIdentity.of("planner-fixture"), "llama", "gguf",
                Map.of(), Map.of("vendor.model", "opaque"), List.of(tensor));

        var safetensors = planner.plan(model, "safetensors");
        var gguf = planner.plan(model, "gguf");

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL,
                safetensors.compatibility().fidelity());
        assertEquals(Set.of("extensions.vendor.model", "tensors.weight.extensions.vendor.scale"),
                safetensors.compatibility().diagnostics().stream()
                        .map(diagnostic -> diagnostic.path().orElseThrow())
                        .collect(java.util.stream.Collectors.toSet()));
        assertEquals(CompatibilityReport.ConversionFidelity.FULL, gguf.compatibility().fidelity());
    }
}
