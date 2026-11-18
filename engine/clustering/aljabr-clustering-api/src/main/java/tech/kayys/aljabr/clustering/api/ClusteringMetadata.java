package tech.kayys.aljabr.clustering.api;

import java.util.Objects;

/**
 * Reproducibility and execution metadata for a completed clustering run.
 *
 * @param algorithmId stable algorithm identifier
 * @param iterations number of completed algorithm iterations or epochs
 * @param seed random seed used by the algorithm
 * @param converged whether the algorithm's configured convergence condition was met
 */
public record ClusteringMetadata(
        String algorithmId,
        int iterations,
        long seed,
        boolean converged) {
    public ClusteringMetadata {
        Objects.requireNonNull(algorithmId, "algorithmId must not be null");
        if (algorithmId.isBlank()) {
            throw new IllegalArgumentException("algorithmId must not be blank");
        }
        if (iterations < 0) {
            throw new IllegalArgumentException("iterations must not be negative");
        }
    }
}
