package tech.kayys.aljabr.gguf.api;

import java.util.Set;

/**
 * Result of comparing required model capabilities with provider capabilities.
 */
public record GgufCapabilityMatch(
        Set<GgufModality> missingInputModalities,
        Set<GgufModality> missingOutputModalities,
        Set<GgufTask> missingTasks) {

    public GgufCapabilityMatch {
        missingInputModalities = missingInputModalities == null
                ? Set.of() : Set.copyOf(missingInputModalities);
        missingOutputModalities = missingOutputModalities == null
                ? Set.of() : Set.copyOf(missingOutputModalities);
        missingTasks = missingTasks == null ? Set.of() : Set.copyOf(missingTasks);
    }

    public boolean compatible() {
        return missingInputModalities.isEmpty()
                && missingOutputModalities.isEmpty()
                && missingTasks.isEmpty();
    }

    /** Human-readable diagnostic suitable for provider selection errors. */
    public String diagnostic() {
        if (compatible()) {
            return "Provider satisfies all required GGUF capabilities";
        }
        return "Provider is missing GGUF capabilities: inputModalities="
                + missingInputModalities + ", outputModalities="
                + missingOutputModalities + ", tasks=" + missingTasks;
    }
}
