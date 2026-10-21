package tech.kayys.aljabr.spi.model;

import java.util.List;
import java.util.Optional;
/**
 * Immutable record representing modelfamilyresolution data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelFamilyResolution(
        Status status,
        String modelType,
        String architectureClassName,
        List<String> familyIds,
        List<ModelFamilySupportReport> supportReports,
        List<ModelFamilyRuntimeManifest> runtimeManifests,
        List<ModelTokenizerDescriptor> tokenizerDescriptors,
        List<String> problemCodes,
        List<String> remediationHints,
        String summary) {/**
 * Enumeration of status values.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core enum operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */


    public enum Status {
        RESOLVED,
        AMBIGUOUS,
        NOT_FOUND
    }

    public ModelFamilyResolution {
        modelType = modelType == null ? "" : modelType;
        architectureClassName = architectureClassName == null ? "" : architectureClassName;
        familyIds = familyIds == null ? List.of() : List.copyOf(familyIds);
        supportReports = supportReports == null ? List.of() : List.copyOf(supportReports);
        runtimeManifests = runtimeManifests == null ? List.of() : List.copyOf(runtimeManifests);
        tokenizerDescriptors = tokenizerDescriptors == null ? List.of() : List.copyOf(tokenizerDescriptors);
        problemCodes = problemCodes == null ? List.of() : List.copyOf(problemCodes);
        remediationHints = remediationHints == null ? List.of() : List.copyOf(remediationHints);
        summary = summary == null ? "" : summary;
    }

    public boolean resolved() {
        return status == Status.RESOLVED;
    }

    public boolean ambiguous() {
        return status == Status.AMBIGUOUS;
    }

    public boolean notFound() {
        return status == Status.NOT_FOUND;
    }

    public Optional<String> primaryFamilyId() {
        return familyIds.isEmpty() ? Optional.empty() : Optional.of(familyIds.get(0));
    }

    public Optional<ModelFamilySupportReport> primarySupportReport() {
        return supportReports.isEmpty() ? Optional.empty() : Optional.of(supportReports.get(0));
    }

    public Optional<ModelFamilyRuntimeManifest> primaryRuntimeManifest() {
        return runtimeManifests.isEmpty() ? Optional.empty() : Optional.of(runtimeManifests.get(0));
    }

    public boolean requiresAttention() {
        return !problemCodes.isEmpty();
    }
}
