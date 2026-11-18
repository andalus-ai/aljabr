package tech.kayys.aljabr.clustering.kmeans;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import tech.kayys.aljabr.clustering.api.ClusteringMetadata;
import tech.kayys.aljabr.clustering.api.ClusteringMetrics;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.core.tensor.Tensor;

/** Owned result of a completed CPU k-means fit. */
final class KMeansResult implements ClusteringResult {
    private final Tensor centroids;
    private final int[] assignments;
    private final ClusteringMetrics metrics;
    private final ClusteringMetadata metadata;
    private final AtomicBoolean closed = new AtomicBoolean();

    KMeansResult(
            Tensor centroids,
            int[] assignments,
            ClusteringMetrics metrics,
            ClusteringMetadata metadata) {
        this.centroids = Objects.requireNonNull(centroids, "centroids must not be null");
        this.assignments = Objects.requireNonNull(assignments, "assignments must not be null");
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
        this.metadata = Objects.requireNonNull(metadata, "metadata must not be null");
    }

    @Override
    public Tensor centroids() {
        ensureOpen();
        return centroids;
    }

    @Override
    public long sampleCount() {
        return assignments.length;
    }

    @Override
    public int clusterCount() {
        return Math.toIntExact(centroids.shape().dim(0));
    }

    @Override
    public int clusterOf(long sampleIndex) {
        ensureOpen();
        if (sampleIndex < 0 || sampleIndex >= assignments.length) {
            throw new IndexOutOfBoundsException("sample index out of range: " + sampleIndex);
        }
        return assignments[Math.toIntExact(sampleIndex)];
    }

    @Override
    public ClusteringMetrics metrics() {
        ensureOpen();
        return metrics;
    }

    @Override
    public ClusteringMetadata metadata() {
        ensureOpen();
        return metadata;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            centroids.close();
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("clustering result is closed");
        }
    }
}
