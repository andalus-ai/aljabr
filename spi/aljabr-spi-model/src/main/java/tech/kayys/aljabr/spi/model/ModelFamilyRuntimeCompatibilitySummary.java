package tech.kayys.aljabr.spi.model;

import java.util.List;
import java.util.Map;
/**
 * Immutable record representing modelfamilyruntimecompatibilitysummary data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelFamilyRuntimeCompatibilitySummary(
        int familyCount,
        List<String> compatibleFamilyIds,
        Map<String, Integer> problemCounts) {

    public ModelFamilyRuntimeCompatibilitySummary {
        compatibleFamilyIds = compatibleFamilyIds == null ? List.of() : List.copyOf(compatibleFamilyIds);
        problemCounts = problemCounts == null ? Map.of() : Map.copyOf(problemCounts);
    }

    public boolean empty() {
        return compatibleFamilyIds.isEmpty() && problemCounts.isEmpty();
    }

    public static ModelFamilyRuntimeCompatibilitySummary emptySummary() {
        return new ModelFamilyRuntimeCompatibilitySummary(0, List.of(), Map.of());
    }
}
