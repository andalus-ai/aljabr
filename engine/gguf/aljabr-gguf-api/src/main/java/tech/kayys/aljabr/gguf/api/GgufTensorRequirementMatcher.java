package tech.kayys.aljabr.gguf.api;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Pure validator for model-family tensor requirements against GGUF inventory. */
public final class GgufTensorRequirementMatcher {

    private GgufTensorRequirementMatcher() {
    }

    public static GgufTensorRequirementMatch match(
            Collection<GgufTensorRequirement> requirements,
            Collection<GgufTensorDescriptor> tensors) {
        Objects.requireNonNull(requirements, "requirements");
        Objects.requireNonNull(tensors, "tensors");

        Map<String, GgufTensorDescriptor> inventory = new HashMap<>();
        for (GgufTensorDescriptor tensor : tensors) {
            Objects.requireNonNull(tensor, "tensor");
            if (inventory.putIfAbsent(tensor.name(), tensor) != null) {
                throw new IllegalArgumentException("Duplicate tensor in GGUF inventory: " + tensor.name());
            }
        }

        Set<String> requiredNames = new HashSet<>();
        List<String> diagnostics = new ArrayList<>();
        for (GgufTensorRequirement requirement : requirements) {
            Objects.requireNonNull(requirement, "requirement");
            if (!requiredNames.add(requirement.name())) {
                throw new IllegalArgumentException("Duplicate tensor requirement: " + requirement.name());
            }
            GgufTensorDescriptor actual = inventory.get(requirement.name());
            if (actual == null) {
                diagnostics.add("Missing required tensor: " + requirement.name());
                continue;
            }
            if (!requirement.dimensions().isEmpty()
                    && !requirement.dimensions().equals(actual.dimensions())) {
                diagnostics.add("Tensor " + requirement.name() + " has dimensions "
                        + actual.dimensions() + "; expected " + requirement.dimensions());
            }
            if (!requirement.acceptedGgmlTypeIds().isEmpty()
                    && !requirement.acceptedGgmlTypeIds().contains(actual.ggmlTypeId())) {
                diagnostics.add("Tensor " + requirement.name() + " has GGML type id "
                        + actual.ggmlTypeId() + "; accepted ids are "
                        + requirement.acceptedGgmlTypeIds());
            }
        }
        return new GgufTensorRequirementMatch(diagnostics);
    }
}
