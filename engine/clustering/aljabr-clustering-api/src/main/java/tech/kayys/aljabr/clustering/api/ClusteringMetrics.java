package tech.kayys.aljabr.clustering.api;

import java.util.Objects;

/**
 * Objective value produced by a clustering run.
 *
 * @param objectiveName stable, human-readable objective identifier
 * @param objectiveValue objective value; lower is better for the named objective
 */
public record ClusteringMetrics(String objectiveName, double objectiveValue) {
    public ClusteringMetrics {
        Objects.requireNonNull(objectiveName, "objectiveName must not be null");
        if (objectiveName.isBlank()) {
            throw new IllegalArgumentException("objectiveName must not be blank");
        }
        if (!Double.isFinite(objectiveValue)) {
            throw new IllegalArgumentException("objectiveValue must be finite");
        }
    }
}
