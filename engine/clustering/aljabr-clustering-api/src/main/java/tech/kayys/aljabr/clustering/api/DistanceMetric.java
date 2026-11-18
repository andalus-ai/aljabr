package tech.kayys.aljabr.clustering.api;

/**
 * Distance functions supported by clustering algorithms.
 *
 * <p>Algorithms may support only a subset. Support must be reported or rejected
 * explicitly; implementations must not silently substitute another metric.</p>
 */
public enum DistanceMetric {
    /** Squared Euclidean distance; the conventional k-means objective. */
    SQUARED_EUCLIDEAN,

    /** Cosine distance, normally used with normalized input vectors. */
    COSINE
}
