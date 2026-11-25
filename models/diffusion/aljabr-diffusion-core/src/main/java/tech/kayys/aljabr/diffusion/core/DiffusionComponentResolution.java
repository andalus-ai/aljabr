package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Immutable result of resolving logical component requirements. */
public record DiffusionComponentResolution(
        String modelId,
        String revision,
        Map<String, DiffusionComponentArtifact> selected,
        List<String> missingRequiredRoles) {

    public DiffusionComponentResolution {
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(revision, "revision");
        selected = Collections.unmodifiableMap(new LinkedHashMap<>(
                Objects.requireNonNull(selected, "selected")));
        missingRequiredRoles = List.copyOf(Objects.requireNonNull(missingRequiredRoles, "missingRequiredRoles"));
    }

    public boolean complete() {
        return missingRequiredRoles.isEmpty();
    }

    public DiffusionComponentArtifact require(String role) {
        DiffusionComponentArtifact artifact = selected.get(role);
        if (artifact == null) {
            throw new IllegalStateException("No resolved diffusion component for role: " + role);
        }
        return artifact;
    }
}
