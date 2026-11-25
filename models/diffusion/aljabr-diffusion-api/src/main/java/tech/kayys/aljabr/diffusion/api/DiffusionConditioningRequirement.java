package tech.kayys.aljabr.diffusion.api;

import java.util.Objects;

/**
 * Declares the cardinality of one conditioning modality accepted by a task.
 *
 * <p>A maximum of {@link Integer#MAX_VALUE} represents an unbounded number
 * of inputs. Requirements describe validation only; they do not prescribe
 * preprocessing or execution.</p>
 */
public record DiffusionConditioningRequirement(
        DiffusionConditioningInput.Modality modality,
        int minimumCount,
        int maximumCount) {

    public DiffusionConditioningRequirement {
        Objects.requireNonNull(modality, "modality");
        if (minimumCount < 0) {
            throw new IllegalArgumentException("minimumCount must be non-negative");
        }
        if (maximumCount < minimumCount) {
            throw new IllegalArgumentException("maximumCount must be >= minimumCount");
        }
    }

    public static DiffusionConditioningRequirement required(
            DiffusionConditioningInput.Modality modality) {
        return new DiffusionConditioningRequirement(modality, 1, Integer.MAX_VALUE);
    }

    public static DiffusionConditioningRequirement optional(
            DiffusionConditioningInput.Modality modality) {
        return new DiffusionConditioningRequirement(modality, 0, Integer.MAX_VALUE);
    }
}
