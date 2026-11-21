package tech.kayys.aljabr.spi.model.ir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * High-level AMIR conversion entry point.
 *
 * <p>The facade coordinates descriptor readers, target capability planners,
 * and conversion executors. It does not implement parsing, compatibility
 * policy, or payload transfer; those remain owned by their respective SPIs.
 */
public final class ModelConversionFacade {

    private final List<ModelDescriptorReader> readers;
    private final ModelConversionPlanner planner;
    private final List<ModelConversionExecutor> executors;

    public ModelConversionFacade(
            List<? extends ModelDescriptorReader> readers,
            ModelConversionPlanner planner,
            List<? extends ModelConversionExecutor> executors) {
        this.readers = List.copyOf(Objects.requireNonNull(readers, "readers must not be null"));
        this.planner = Objects.requireNonNull(planner, "planner must not be null");
        this.executors = List.copyOf(Objects.requireNonNull(executors, "executors must not be null"));
    }

    /** Creates a facade from providers available through ServiceLoader. */
    public static ModelConversionFacade discover() {
        return new ModelConversionFacade(
                ModelDescriptorReaders.discover(),
                ModelConversionPlanners.discover(),
                ModelConversionExecutors.discover());
    }

    /**
     * Reads source metadata and produces a compatibility plan. No conversion
     * executor is selected and no output is written by this method.
     */
    public PreparedConversion prepare(Path source, String sourceFormat, String targetFormat)
            throws IOException {
        Objects.requireNonNull(source, "source must not be null");
        requireText(sourceFormat, "sourceFormat");
        requireText(targetFormat, "targetFormat");

        ModelDescriptorReader reader = ModelDescriptorReaders.find(readers, sourceFormat)
                .orElseThrow(() -> new IllegalStateException(
                        "No AMIR descriptor reader is registered for source format " + sourceFormat));
        ModelDescriptor model = reader.read(source);
        if (!model.sourceFormat().equalsIgnoreCase(sourceFormat)) {
            throw new IOException("Descriptor reader returned source format '"
                    + model.sourceFormat() + "' for requested format '" + sourceFormat + "'");
        }
        return new PreparedConversion(model, planner.plan(model, targetFormat));
    }

    /**
     * Executes a previously prepared conversion using the unique executor for
     * its source/target pair.
     *
     * @throws IOException when the selected executor cannot complete the write
     * @throws IllegalStateException when the plan is unsupported, partial
     *         fidelity is not acknowledged, or no unique executor exists
     */
    public ConversionExecutionResult execute(
            PreparedConversion prepared, ConversionRequest request) throws IOException {
        Objects.requireNonNull(prepared, "prepared must not be null");
        Objects.requireNonNull(request, "request must not be null");

        ConversionPlan plan = prepared.plan();
        if (!plan.isConvertible()) {
            throw new IllegalStateException("Conversion is unsupported: "
                    + plan.compatibility().diagnostics());
        }
        if (plan.requiresAcknowledgement() && !request.acknowledgePartial()) {
            throw new IllegalStateException(
                    "Partial-fidelity conversion requires explicit acknowledgement");
        }

        Optional<ModelConversionExecutor> selected = ModelConversionExecutors.find(
                executors, plan.sourceFormat(), plan.targetFormat());
        ModelConversionExecutor executor = selected.orElseThrow(() ->
                new IllegalStateException("No conversion executor is registered for "
                        + plan.sourceFormat() + " -> " + plan.targetFormat()));

        return executor.execute(prepared.model(), plan, request);
    }

    /** Convenience method that prepares and executes a conversion in one call. */
    public ConversionExecutionResult convert(
            Path source,
            String sourceFormat,
            String targetFormat,
            ConversionRequest request) throws IOException {
        return execute(prepare(source, sourceFormat, targetFormat), request);
    }

    private static void requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
    }
}
