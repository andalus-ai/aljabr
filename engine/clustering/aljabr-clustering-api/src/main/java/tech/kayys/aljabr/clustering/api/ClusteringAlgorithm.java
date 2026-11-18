package tech.kayys.aljabr.clustering.api;

/**
 * Fits a clustering model to a vector dataset.
 *
 * @param <C> immutable configuration type accepted by this algorithm
 */
public interface ClusteringAlgorithm<C extends ClusteringConfig> {
    /** Stable identifier, for example {@code kmeans} or {@code emeans}. */
    String id();

    ClusteringResult fit(ClusteringDataset dataset, C config);
}
