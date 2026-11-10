package tech.kayys.aljabr.spi.model;

import java.util.List;
import java.util.Map;

/**
 * Machine-readable support report for a {@link ModelFamilyPlugin}.
 *
 * <p>
 * The support report is a flattened, automation-friendly snapshot of a plugin's
 * surface at a point in time. It must stay consistent with the plugin's
 * {@link ModelFamilyDescriptor}, adapter list, and tokenizer descriptors;
 * drift is detected by {@link ModelFamilyContractValidator}.
 *
 * @param id                      family identifier (must match descriptor)
 * @param displayName             human-readable name (must match descriptor)
 * @param modelTypes              model-type identifiers (must match descriptor)
 * @param architectures           architecture class names from adapters
 * @param architectureAdapterIds  adapter IDs returned by the plugin
 * @param tokenizerIds            tokenizer descriptor IDs
 * @param tokenizerKinds          tokenizer algorithm kinds
 * @param bundleProfile           resolved bundle profile tier
 * @param capabilities            declared capabilities (must match descriptor)
 * @param directSafetensorStatus  readiness of direct SafeTensor inference
 * @param directSafetensorReason  optional custom reason/path for direct
 *                                SafeTensor status
 * @param directSafetensorCaveats extra runtime requirement metadata keyed by
 *                                scope
 * @param metadata                additional freeform key-value pairs
 * @author bhangun
 */
public record ModelFamilySupportReport(
        String familyId,
        String displayName,
        List<String> modelTypes,
        List<String> architectureClassNames,
        List<String> architectureAdapterIds,
        List<String> tokenizerProfileIds,
        List<ModelTokenizerKind> tokenizerKinds,
        ModelFamilyBundleProfile bundleProfile,
        List<ModelFamilyCapability> capabilities,
        ModelFamilyDirectSupport directSafetensorStatus,
        String directSafetensorReason,
        Map<String, String> directSafetensorCaveats,
        Map<String, String> metadata) {

    public ModelFamilySupportReport {
        familyId = familyId == null ? "" : familyId;
        displayName = displayName == null ? "" : displayName;
        modelTypes = modelTypes == null ? List.of() : List.copyOf(modelTypes);
        architectureClassNames = architectureClassNames == null ? List.of() : List.copyOf(architectureClassNames);
        architectureAdapterIds = architectureAdapterIds == null ? List.of() : List.copyOf(architectureAdapterIds);
        tokenizerProfileIds = tokenizerProfileIds == null ? List.of() : List.copyOf(tokenizerProfileIds);
        tokenizerKinds = tokenizerKinds == null ? List.of() : List.copyOf(tokenizerKinds);
        capabilities = capabilities == null ? List.of() : List.copyOf(capabilities);
        directSafetensorReason = directSafetensorReason == null ? "" : directSafetensorReason;
        directSafetensorCaveats = directSafetensorCaveats == null ? Map.of() : Map.copyOf(directSafetensorCaveats);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public String id() {
        return familyId;
    }

    public boolean directSafetensorReady() {
        return directSafetensorStatus == ModelFamilyDirectSupport.READY;
    }

    public String shortDirectSafetensorSummary() {
        if (directSafetensorStatus == ModelFamilyDirectSupport.READY)
            return "ready";
        String summary = directSafetensorStatus == ModelFamilyDirectSupport.EXPERIMENTAL
                ? "experimental:" + directSafetensorReason
                : directSafetensorReason;
        if (directSafetensorCaveats != null && !directSafetensorCaveats.isEmpty()) {
            String c = directSafetensorCaveats.entrySet().stream()
                    .sorted(java.util.Map.Entry.comparingByKey())
                    .map(e -> e.getKey() + ":" + e.getValue())
                    .collect(java.util.stream.Collectors.joining(","));
            summary += ";caveats=" + c;
        }
        return summary;
    }

    public boolean defaultBundle() {
        return bundleProfile == ModelFamilyBundleProfile.CORE;
    }

}
