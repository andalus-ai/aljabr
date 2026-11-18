package tech.kayys.aljabr.clustering.api;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Result of fitting a clustering algorithm.
 *
 * <p>The result owns its centroid tensor and must be closed by the caller.
 * The tensor returned by {@link #centroids()} is borrowed and remains valid
 * only until this result is closed. Assignment lookup is long-indexed so the
 * public contract is not restricted to Java array-sized datasets.</p>
 */
public interface ClusteringResult extends AutoCloseable {
    /** Returns the owned [clusterCount, dimensions] centroid tensor as a borrow. */
    Tensor centroids();

    long sampleCount();

    int clusterCount();

    /** Returns the assigned cluster index for the given sample. */
    int clusterOf(long sampleIndex);

    ClusteringMetrics metrics();

    ClusteringMetadata metadata();

    @Override
    void close();
}
