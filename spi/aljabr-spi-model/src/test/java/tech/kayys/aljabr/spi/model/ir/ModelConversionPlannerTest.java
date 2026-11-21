package tech.kayys.aljabr.spi.model.ir;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelConversionPlannerTest {

    private final ModelDescriptor source = new ModelDescriptor(
            ModelIdentity.of("sample"),
            "llama",
            "safetensors",
            Map.of(),
            Map.of(),
            List.of());

    @Test
    void returnsFullPlanWhenTargetReportsFullFidelity() {
        var planner = new ModelConversionPlanner(List.of(capability(
                "GGUF", CompatibilityReport.ConversionFidelity.FULL)));

        var plan = planner.plan(source, "gguf");

        assertTrue(plan.isConvertible());
        assertTrue(plan.isLossless());
        assertFalse(plan.requiresAcknowledgement());
        assertEquals("gguf", plan.targetFormat());
    }

    @Test
    void requiresAcknowledgementForPartialFidelity() {
        var planner = new ModelConversionPlanner(List.of(capability(
                "gguf", CompatibilityReport.ConversionFidelity.PARTIAL)));

        var plan = planner.plan(source, "GGUF");

        assertTrue(plan.isConvertible());
        assertFalse(plan.isLossless());
        assertTrue(plan.requiresAcknowledgement());
    }

    @Test
    void unknownTargetIsExplicitlyUnsupported() {
        var planner = new ModelConversionPlanner(List.of());

        var plan = planner.plan(source, "onnx");

        assertFalse(plan.isConvertible());
        assertEquals(CompatibilityReport.ConversionFidelity.UNSUPPORTED,
                plan.compatibility().fidelity());
        assertEquals("TARGET_FORMAT_UNAVAILABLE",
                plan.compatibility().diagnostics().getFirst().code());
    }

    @Test
    void rejectsDuplicateTargetProvidersIgnoringCase() {
        assertThrows(IllegalArgumentException.class, () -> new ModelConversionPlanner(List.of(
                capability("gguf", CompatibilityReport.ConversionFidelity.FULL),
                capability("GGUF", CompatibilityReport.ConversionFidelity.FULL))));
    }

    private static TargetFormatCapability capability(
            String format, CompatibilityReport.ConversionFidelity fidelity) {
        return new TargetFormatCapability() {
            @Override
            public String targetFormat() {
                return format;
            }

            @Override
            public CompatibilityReport assess(ModelDescriptor ignored) {
                return new CompatibilityReport(fidelity, List.of());
            }
        };
    }
}
