package tech.kayys.aljabr.gguf.api;

import java.util.EnumSet;
import java.util.Set;

/**
 * Pure capability matching logic shared by model adapters and GGUF providers.
 */
public final class GgufCapabilityMatcher {

    private GgufCapabilityMatcher() {
    }

    /**
     * Compares requirements against capabilities actually implemented by a provider.
     *
     * @param required capabilities required by the model/task
     * @param available capabilities implemented by the provider
     */
    public static GgufCapabilityMatch match(
            GgufCapabilitySet required, GgufCapabilitySet available) {
        if (required == null) {
            throw new IllegalArgumentException("required capabilities must not be null");
        }
        if (available == null) {
            throw new IllegalArgumentException("available capabilities must not be null");
        }
        return new GgufCapabilityMatch(
                missing(required.inputModalities(), available.inputModalities()),
                missing(required.outputModalities(), available.outputModalities()),
                missing(required.tasks(), available.tasks()));
    }

    private static <E extends Enum<E>> Set<E> missing(Set<E> required, Set<E> available) {
        if (required.isEmpty()) {
            return Set.of();
        }
        EnumSet<E> result = EnumSet.noneOf(required.iterator().next().getDeclaringClass());
        result.addAll(required);
        result.removeAll(available);
        return Set.copyOf(result);
    }
}
