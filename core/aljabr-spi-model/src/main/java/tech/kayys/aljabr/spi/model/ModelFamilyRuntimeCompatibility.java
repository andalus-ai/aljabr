package tech.kayys.aljabr.spi.model;

import java.util.List;
/**
 * Immutable record representing modelfamilyruntimecompatibility data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelFamilyRuntimeCompatibility(
        boolean compatible,
        String selectedArchitectureAdapterId,
        String selectedArchitectureAdapterBy,
        List<String> problemCodes,
        List<String> remediationHints,
        ModelFamilyResolution modelFamily) {

    public ModelFamilyRuntimeCompatibility {
        selectedArchitectureAdapterId = selectedArchitectureAdapterId == null ? "" : selectedArchitectureAdapterId;
        selectedArchitectureAdapterBy = selectedArchitectureAdapterBy == null ? "" : selectedArchitectureAdapterBy;
        problemCodes = problemCodes == null ? List.of() : List.copyOf(problemCodes);
        remediationHints = remediationHints == null ? List.of() : List.copyOf(remediationHints);
    }

    public List<String> architectureAdapterIds() {
        return selectedArchitectureAdapterId == null || selectedArchitectureAdapterId.isEmpty()
                ? List.of()
                : List.of(selectedArchitectureAdapterId);
    }
}
