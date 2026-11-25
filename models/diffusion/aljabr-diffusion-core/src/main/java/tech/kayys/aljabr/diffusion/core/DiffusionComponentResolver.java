package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentManifest;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentRequirement;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves a capability's logical component requirements against a model
 * manifest without knowing the model family, storage format implementation,
 * or execution backend.
 *
 * <p>When multiple compatible artifacts exist for a role, the preferred
 * variant wins; otherwise declaration order is preserved. This keeps variant
 * choice deterministic and leaves actual artifact loading to format adapters.</p>
 */
public final class DiffusionComponentResolver {

    public DiffusionComponentResolution resolve(
            DiffusionComponentManifest manifest,
            List<DiffusionComponentRequirement> requirements) {
        Objects.requireNonNull(manifest, "manifest");
        Objects.requireNonNull(requirements, "requirements");

        Map<String, DiffusionComponentArtifact> selected = new LinkedHashMap<>();
        List<String> missing = new ArrayList<>();

        for (DiffusionComponentRequirement requirement : requirements) {
            Objects.requireNonNull(requirement, "requirement");
            List<DiffusionComponentArtifact> candidates = manifest.componentsWithRole(requirement.role())
                    .stream()
                    .filter(artifact -> requirement.acceptedFormats().isEmpty()
                            || requirement.acceptedFormats().contains(artifact.format()))
                    .toList();

            DiffusionComponentArtifact chosen = candidates.stream()
                    .filter(artifact -> requirement.preferredVariant() != null
                            && requirement.preferredVariant().equals(artifact.variant()))
                    .findFirst()
                    .orElseGet(() -> candidates.isEmpty() ? null : candidates.get(0));

            if (chosen == null) {
                if (requirement.required()) missing.add(requirement.role());
            } else {
                selected.put(requirement.role(), chosen);
            }
        }

        return new DiffusionComponentResolution(manifest.modelId(), manifest.revision(), selected, missing);
    }
}
