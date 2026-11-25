package tech.kayys.aljabr.diffusion.api;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Ordered, immutable collection of request conditioning inputs.
 *
 * <p>Input ordering is preserved. Video frame sequence indices must be unique;
 * callers remain responsible for supplying the intended temporal order.</p>
 */
public record DiffusionConditioningSet(List<DiffusionConditioningInput> inputs) {

    public DiffusionConditioningSet {
        inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs"));
        Set<String> ids = new HashSet<>();
        Set<Integer> frameIndices = new HashSet<>();
        for (DiffusionConditioningInput input : inputs) {
            Objects.requireNonNull(input, "input");
            if (!ids.add(input.id())) {
                throw new IllegalArgumentException("duplicate conditioning input id: " + input.id());
            }
            if (input.modality() == DiffusionConditioningInput.Modality.VIDEO_FRAME
                    && !frameIndices.add(input.sequenceIndex())) {
                throw new IllegalArgumentException(
                        "duplicate video frame sequence index: " + input.sequenceIndex());
            }
        }
    }

    public List<DiffusionConditioningInput> inputsOf(DiffusionConditioningInput.Modality modality) {
        Objects.requireNonNull(modality, "modality");
        return inputs.stream().filter(input -> input.modality() == modality).toList();
    }

    /**
     * Validates this set against a task's declared modality cardinalities.
     * Inputs for modalities not declared by the requirements are rejected.
     */
    public void validateAgainst(List<DiffusionConditioningRequirement> requirements) {
        Objects.requireNonNull(requirements, "requirements");
        Set<DiffusionConditioningInput.Modality> declared = new HashSet<>();
        for (DiffusionConditioningRequirement requirement : requirements) {
            Objects.requireNonNull(requirement, "requirement");
            if (!declared.add(requirement.modality())) {
                throw new IllegalArgumentException(
                        "duplicate conditioning requirement: " + requirement.modality());
            }
            long count = inputs.stream()
                    .filter(input -> input.modality() == requirement.modality())
                    .count();
            if (count < requirement.minimumCount() || count > requirement.maximumCount()) {
                throw new IllegalArgumentException(
                        "Conditioning modality " + requirement.modality() + " requires "
                                + requirement.minimumCount() + ".." + requirement.maximumCount()
                                + " inputs but received " + count);
            }
        }
        for (DiffusionConditioningInput input : inputs) {
            if (!declared.contains(input.modality())) {
                throw new IllegalArgumentException(
                        "Conditioning modality is not declared: " + input.modality());
            }
        }
    }

    public boolean isEmpty() {
        return inputs.isEmpty();
    }
}
