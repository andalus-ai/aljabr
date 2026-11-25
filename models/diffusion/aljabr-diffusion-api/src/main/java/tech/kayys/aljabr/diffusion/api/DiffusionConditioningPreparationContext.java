package tech.kayys.aljabr.diffusion.api;

import java.util.Map;
import java.util.Objects;

/**
 * Immutable, caller-provided options for conditioning preparation.
 *
 * <p>The context deliberately contains no device, tensor, image, video or
 * tokenizer types. Integrations may define namespaced string options without
 * coupling the generic diffusion API to a particular preprocessing library.</p>
 */
public record DiffusionConditioningPreparationContext(Map<String, String> options) {

    public DiffusionConditioningPreparationContext {
        options = Map.copyOf(Objects.requireNonNull(options, "options"));
        options.forEach((key, value) -> {
            if (key == null || key.isBlank()) {
                throw new IllegalArgumentException("option keys must not be blank");
            }
            Objects.requireNonNull(value, "option value");
        });
    }

    public static DiffusionConditioningPreparationContext defaults() {
        return new DiffusionConditioningPreparationContext(Map.of());
    }
}
