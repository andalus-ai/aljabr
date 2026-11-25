package tech.kayys.aljabr.diffusion.api;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable inventory of the independently loadable artifacts for a model
 * revision. It describes files and their semantic roles, not their execution.
 */
public record DiffusionComponentManifest(
        String modelId,
        String revision,
        List<DiffusionComponentArtifact> components) {

    public DiffusionComponentManifest {
        modelId = requireText(modelId, "modelId");
        revision = requireText(revision, "revision");
        components = List.copyOf(Objects.requireNonNull(components, "components"));
        Set<String> ids = new HashSet<>();
        for (DiffusionComponentArtifact component : components) {
            Objects.requireNonNull(component, "component");
            if (!ids.add(component.id())) {
                throw new IllegalArgumentException("duplicate component id: " + component.id());
            }
        }
    }

    public List<DiffusionComponentArtifact> componentsWithRole(String role) {
        String requiredRole = requireText(role, "role");
        return components.stream().filter(component -> component.role().equals(requiredRole)).toList();
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
        return value;
    }
}
