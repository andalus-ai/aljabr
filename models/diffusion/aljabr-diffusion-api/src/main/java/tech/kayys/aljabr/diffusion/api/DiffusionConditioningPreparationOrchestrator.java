package tech.kayys.aljabr.diffusion.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Validates and prepares a complete conditioning set using registered
 * modality/source adapters.
 *
 * <p>Task requirements are checked before any adapter is invoked. Inputs are
 * prepared in their declared order. The returned set owns all prepared
 * resources; if preparation fails, resources already acquired are closed in
 * reverse order and cleanup failures are suppressed on the original failure.</p>
 */
public final class DiffusionConditioningPreparationOrchestrator {

    private final DiffusionConditioningPreprocessorRegistry registry;

    public DiffusionConditioningPreparationOrchestrator(
            DiffusionConditioningPreprocessorRegistry registry) {
        this.registry = Objects.requireNonNull(registry, "registry");
    }

    public PreparedDiffusionConditioningSet prepare(
            DiffusionConditioningSet inputs,
            List<DiffusionConditioningRequirement> requirements,
            DiffusionConditioningPreparationContext context) {
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(context, "context");

        // Fail fast before acquiring any adapter-owned resource.
        inputs.validateAgainst(requirements);

        var prepared = new ArrayList<DiffusionConditioningData>(inputs.inputs().size());
        try {
            for (var input : inputs.inputs()) {
                var preprocessor = registry.resolve(input);
                var data = preprocessor.prepare(input, context);
                if (data == null) {
                    throw new DiffusionConditioningPreparationException(
                            input, "Preprocessor '" + preprocessor.id()
                                    + "' returned null prepared data");
                }
                if (!input.id().equals(data.inputId()) || input.modality() != data.modality()) {
                    var mismatch = new DiffusionConditioningPreparationException(
                            input, "Preprocessor '" + preprocessor.id()
                                    + "' returned data for a different input");
                    try {
                        data.close();
                    } catch (Exception closeFailure) {
                        mismatch.addSuppressed(closeFailure);
                    }
                    throw mismatch;
                }
                prepared.add(data);
            }
            return new PreparedDiffusionConditioningSet(prepared);
        } catch (Throwable failure) {
            closeAfterFailure(prepared, failure);
            throw failure;
        }
    }

    private static void closeAfterFailure(
            List<DiffusionConditioningData> prepared, Throwable failure) {
        for (int i = prepared.size() - 1; i >= 0; i--) {
            try {
                prepared.get(i).close();
            } catch (Throwable closeFailure) {
                if (closeFailure != failure) {
                    failure.addSuppressed(closeFailure);
                }
            }
        }
    }
}
