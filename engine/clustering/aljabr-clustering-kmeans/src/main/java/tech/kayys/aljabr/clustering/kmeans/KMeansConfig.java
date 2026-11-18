package tech.kayys.aljabr.clustering.kmeans;

import java.util.Objects;

import tech.kayys.aljabr.clustering.api.ClusteringConfig;
import tech.kayys.aljabr.clustering.api.DistanceMetric;

/** Immutable configuration for Lloyd's batch k-means algorithm. */
public record KMeansConfig(
        int clusterCount,
        long seed,
        DistanceMetric distanceMetric,
        int maxIterations,
        double tolerance,
        KMeansInitialization initialization) implements ClusteringConfig {

    public KMeansConfig {
        if (clusterCount < 1) {
            throw new IllegalArgumentException("clusterCount must be positive");
        }
        Objects.requireNonNull(distanceMetric, "distanceMetric must not be null");
        if (maxIterations < 1) {
            throw new IllegalArgumentException("maxIterations must be positive");
        }
        if (!Double.isFinite(tolerance) || tolerance < 0.0) {
            throw new IllegalArgumentException("tolerance must be finite and non-negative");
        }
        Objects.requireNonNull(initialization, "initialization must not be null");
    }

    public static KMeansConfig defaults(int clusterCount, long seed) {
        return new KMeansConfig(
                clusterCount, seed, DistanceMetric.SQUARED_EUCLIDEAN,
                300, 1.0e-4, KMeansInitialization.KMEANS_PLUS_PLUS);
    }
}
