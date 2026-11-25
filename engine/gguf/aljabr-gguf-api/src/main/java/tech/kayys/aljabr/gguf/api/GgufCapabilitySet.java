package tech.kayys.aljabr.gguf.api;

import java.util.Set;

/**
 * Immutable set of modality and task capabilities.
 *
 * <p>The same value type can describe model-declared capabilities and
 * provider-implemented capabilities. Keep those profiles separate when
 * evaluating compatibility: a model declaration is not proof that a
 * provider can execute it.</p>
 */
public record GgufCapabilitySet(
        Set<GgufModality> inputModalities,
        Set<GgufModality> outputModalities,
        Set<GgufTask> tasks) {

    public GgufCapabilitySet {
        inputModalities = inputModalities == null ? Set.of() : Set.copyOf(inputModalities);
        outputModalities = outputModalities == null ? Set.of() : Set.copyOf(outputModalities);
        tasks = tasks == null ? Set.of() : Set.copyOf(tasks);
    }

    public static GgufCapabilitySet textGeneration() {
        return new GgufCapabilitySet(
                Set.of(GgufModality.TEXT),
                Set.of(GgufModality.TEXT),
                Set.of(GgufTask.TEXT_GENERATION));
    }

    public static GgufCapabilitySet none() {
        return new GgufCapabilitySet(Set.of(), Set.of(), Set.of());
    }

    public GgufCapabilityMatch supports(GgufCapabilitySet required) {
        return GgufCapabilityMatcher.match(required, this);
    }
}
