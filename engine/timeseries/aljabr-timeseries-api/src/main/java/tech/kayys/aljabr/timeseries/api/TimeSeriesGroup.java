package tech.kayys.aljabr.timeseries.api;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * A forecasting task's related targets and optional covariates.
 *
 * <p>Targets in a group are jointly visible to models that support
 * cross-series learning. Group boundaries are semantic isolation boundaries:
 * a forecasting model must not mix information between different groups.
 */
public record TimeSeriesGroup(
        String id,
        List<TimeSeries> targets,
        List<CovariateSeries> covariates) {

    public TimeSeriesGroup {
        Objects.requireNonNull(id, "id");
        if (id.isBlank()) throw new IllegalArgumentException("group id must not be blank");
        targets = List.copyOf(Objects.requireNonNull(targets, "targets"));
        covariates = covariates == null ? List.of() : List.copyOf(covariates);
        if (targets.isEmpty()) throw new IllegalArgumentException("a group must contain at least one target");
        Set<String> targetIds = new HashSet<>();
        for (TimeSeries target : targets) {
            Objects.requireNonNull(target, "target");
            if (!targetIds.add(target.id())) throw new IllegalArgumentException("duplicate target id: " + target.id());
        }
        Set<String> covariateNames = new HashSet<>();
        for (CovariateSeries covariate : covariates) {
            Objects.requireNonNull(covariate, "covariate");
            if (!covariateNames.add(covariate.name())) {
                throw new IllegalArgumentException("duplicate covariate name: " + covariate.name());
            }
        }
    }
}
