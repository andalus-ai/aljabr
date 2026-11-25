package tech.kayys.aljabr.diffusion.api;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable registry for conditioning preprocessors.
 *
 * <p>Selection is capability-based. An input must have exactly one matching
 * preprocessor; the registry never silently chooses between competing
 * adapters based on registration order.</p>
 */
public final class DiffusionConditioningPreprocessorRegistry {

    private final List<DiffusionConditioningPreprocessor> preprocessors;
    private final Map<String, DiffusionConditioningPreprocessor> byId;

    public DiffusionConditioningPreprocessorRegistry(
            List<? extends DiffusionConditioningPreprocessor> preprocessors) {
        Objects.requireNonNull(preprocessors, "preprocessors");
        var ordered = new ArrayList<DiffusionConditioningPreprocessor>(preprocessors.size());
        var identifiers = new LinkedHashMap<String, DiffusionConditioningPreprocessor>();
        for (var preprocessor : preprocessors) {
            Objects.requireNonNull(preprocessor, "preprocessor");
            String id = Objects.requireNonNull(preprocessor.id(), "preprocessor id");
            if (id.isBlank()) {
                throw new IllegalArgumentException("preprocessor id must not be blank");
            }
            if (identifiers.putIfAbsent(id, preprocessor) != null) {
                throw new IllegalArgumentException("duplicate conditioning preprocessor id: " + id);
            }
            ordered.add(preprocessor);
        }
        this.preprocessors = List.copyOf(ordered);
        this.byId = Map.copyOf(identifiers);
    }

    /** Returns registered adapters in their original registration order. */
    public List<DiffusionConditioningPreprocessor> preprocessors() {
        return preprocessors;
    }

    /**
     * Resolves exactly one adapter for the input.
     *
     * @throws DiffusionConditioningPreparationException if no adapter or
     *         multiple adapters support the input
     */
    public DiffusionConditioningPreprocessor resolve(DiffusionConditioningInput input) {
        Objects.requireNonNull(input, "input");
        DiffusionConditioningPreprocessor match = null;
        int matches = 0;
        for (var candidate : preprocessors) {
            if (candidate.supports(input)) {
                match = candidate;
                matches++;
            }
        }
        if (matches == 0) {
            throw new DiffusionConditioningPreparationException(
                    input, "No conditioning preprocessor supports input '" + input.id()
                            + "' (" + input.modality() + ")");
        }
        if (matches > 1) {
            throw new DiffusionConditioningPreparationException(
                    input, "Ambiguous conditioning preprocessors for input '" + input.id()
                            + "': " + matches + " adapters match");
        }
        return match;
    }
}
