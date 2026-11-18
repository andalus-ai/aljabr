package tech.kayys.aljabr.clustering.kmeans;

/** Centroid seeding strategies supported by the Lloyd k-means implementation. */
public enum KMeansInitialization {
    /** Select distinct input rows uniformly at random. */
    RANDOM,
    /** Select seeds with probability proportional to squared distance from the nearest seed. */
    KMEANS_PLUS_PLUS
}
