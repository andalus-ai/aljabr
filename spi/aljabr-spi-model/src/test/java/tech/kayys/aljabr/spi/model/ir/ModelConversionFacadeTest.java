package tech.kayys.aljabr.spi.model.ir;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class ModelConversionFacadeTest {

    private static final ModelDescriptor MODEL = new ModelDescriptor(
            ModelIdentity.of("sample"), "llama", "source", Map.of(), Map.of(), List.of());

    @Test
    void preparesAndExecutesThroughTheSelectedProviders() throws IOException {
        AtomicBoolean executed = new AtomicBoolean();
        ModelConversionFacade facade = facade(
                CompatibilityReport.ConversionFidelity.FULL,
                executor(executed));

        PreparedConversion prepared = facade.prepare(Path.of("sample.model"), "source", "target");
        assertEquals(MODEL.identity(), prepared.plan().sourceIdentity());
        assertFalse(executed.get());

        ConversionExecutionResult result = facade.execute(
                prepared, ConversionRequest.to(URI.create("file:///tmp/output.model")));

        assertTrue(executed.get());
        assertEquals("target", result.targetFormat());
        assertEquals(3, result.tensorCount());
    }

    @Test
    void partialPlanRequiresExplicitAcknowledgementBeforeExecutorRuns() throws IOException {
        AtomicBoolean executed = new AtomicBoolean();
        ModelConversionFacade facade = facade(
                CompatibilityReport.ConversionFidelity.PARTIAL,
                executor(executed));
        PreparedConversion prepared = facade.prepare(Path.of("sample.model"), "source", "target");

        assertThrows(IllegalStateException.class, () -> facade.execute(
                prepared, ConversionRequest.to(URI.create("file:///tmp/output.model"))));
        assertFalse(executed.get());

        facade.execute(prepared, new ConversionRequest(
                URI.create("file:///tmp/output.model"), true, Map.of()));
        assertTrue(executed.get());
    }

    @Test
    void unsupportedPlanNeverInvokesExecutor() throws IOException {
        AtomicBoolean executed = new AtomicBoolean();
        ModelConversionFacade facade = facade(
                CompatibilityReport.ConversionFidelity.UNSUPPORTED,
                executor(executed));
        PreparedConversion prepared = facade.prepare(Path.of("sample.model"), "source", "target");

        assertThrows(IllegalStateException.class, () -> facade.execute(
                prepared, new ConversionRequest(
                        URI.create("file:///tmp/output.model"), true, Map.of())));
        assertFalse(executed.get());
    }

    private static ModelConversionFacade facade(
            CompatibilityReport.ConversionFidelity fidelity,
            ModelConversionExecutor executor) {
        ModelDescriptorReader reader = new ModelDescriptorReader() {
            @Override
            public java.util.Set<String> formats() {
                return java.util.Set.of("source");
            }

            @Override
            public ModelDescriptor read(Path ignored) {
                return MODEL;
            }
        };
        TargetFormatCapability capability = new TargetFormatCapability() {
            @Override
            public String targetFormat() {
                return "target";
            }

            @Override
            public CompatibilityReport assess(ModelDescriptor ignored) {
                return new CompatibilityReport(fidelity, List.of());
            }
        };
        return new ModelConversionFacade(
                List.of(reader), new ModelConversionPlanner(List.of(capability)), List.of(executor));
    }

    private static ModelConversionExecutor executor(AtomicBoolean executed) {
        return new ModelConversionExecutor() {
            @Override
            public boolean supports(String sourceFormat, String targetFormat) {
                return sourceFormat.equalsIgnoreCase("source")
                        && targetFormat.equalsIgnoreCase("target");
            }

            @Override
            public ConversionExecutionResult execute(
                    ModelDescriptor model, ConversionPlan plan, ConversionRequest request) {
                executed.set(true);
                return new ConversionExecutionResult(
                        plan.sourceFormat(), plan.targetFormat(), request.output(),
                        3, 24L, plan.compatibility(), List.of());
            }
        };
    }
}
