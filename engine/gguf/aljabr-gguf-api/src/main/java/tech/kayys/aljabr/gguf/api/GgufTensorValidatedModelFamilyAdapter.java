package tech.kayys.aljabr.gguf.api;

import java.util.Collection;
import java.util.Collections;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Reusable model-family adapter base for families whose artifact contract can
 * be expressed as architecture identifiers and named tensor requirements.
 *
 * <p>Implementations provide verified architecture identifiers and tensor
 * requirements. The default inspection checks those declarations against the
 * parsed artifact without loading tensor payloads or inferring runtime support.</p>
 */
public interface GgufTensorValidatedModelFamilyAdapter extends GgufModelFamilyAdapter {

    /** Exact GGUF architecture identifiers supported by this adapter. */
    Set<String> supportedArchitectures();

    /** Required tensor names and optional exact shape/type constraints. */
    Collection<GgufTensorRequirement> tensorRequirements();

    @Override
    default GgufModelFamilyInspection inspect(GgufModelDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        Set<String> architectures = normalizedArchitectures();
        String actualArchitecture = descriptor.architecture();
        if (actualArchitecture == null || !architectures.contains(actualArchitecture)) {
            return GgufModelFamilyInspection.rejected(
                    familyId(),
                    GgufModelFamilyInspection.Status.UNSUPPORTED_ARCHITECTURE,
                    java.util.List.of("Unsupported GGUF architecture: "
                            + (actualArchitecture == null ? "missing" : actualArchitecture)
                            + "; supported architectures are " + architectures));
        }

        Collection<GgufTensorRequirement> requirements =
                Objects.requireNonNull(tensorRequirements(), "tensorRequirements");
        GgufTensorRequirementMatch match = GgufTensorRequirementMatcher.match(
                requirements, descriptor.tensorDescriptors().values());
        if (!match.compatible()) {
            return GgufModelFamilyInspection.rejected(
                    familyId(),
                    GgufModelFamilyInspection.Status.INCOMPLETE_ARTIFACT,
                    match.diagnostics());
        }
        return GgufModelFamilyInspection.match(familyId());
    }

    private Set<String> normalizedArchitectures() {
        Set<String> declared = Objects.requireNonNull(
                supportedArchitectures(), "supportedArchitectures");
        TreeSet<String> normalized = new TreeSet<>();
        for (String architecture : declared) {
            Objects.requireNonNull(architecture, "supported architecture");
            String value = architecture.trim();
            if (value.isEmpty()) {
                throw new IllegalStateException("supported architectures must not contain blanks");
            }
            normalized.add(value);
        }
        if (normalized.isEmpty()) {
            throw new IllegalStateException("at least one supported architecture is required");
        }
        return Collections.unmodifiableSet(normalized);
    }
}
