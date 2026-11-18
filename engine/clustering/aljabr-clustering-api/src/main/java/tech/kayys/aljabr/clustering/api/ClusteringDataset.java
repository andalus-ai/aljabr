package tech.kayys.aljabr.clustering.api;

import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * A two-dimensional collection of vectors represented by an Aljabr tensor.
 *
 * <p>The tensor must have shape {@code [sampleCount, dimensions]}. This wrapper
 * borrows the supplied tensor: it neither copies nor closes it. The caller must
 * keep the tensor alive for the duration of any clustering operation.</p>
 *
 * <p>The API deliberately does not impose a model format, storage layout, device,
 * or floating-point precision. An algorithm must validate the combinations it
 * actually supports and fail explicitly when it cannot execute them.</p>
 */
public final class ClusteringDataset {
    private final Tensor vectors;

    public ClusteringDataset(Tensor vectors) {
        this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
        if (vectors.shape().rank() != 2) {
            throw new IllegalArgumentException(
                    "Clustering vectors must be rank 2 [samples, dimensions], got " + vectors.shape());
        }
        if (vectors.shape().dim(0) == 0) {
            throw new IllegalArgumentException("Clustering dataset must contain at least one sample");
        }
        if (vectors.shape().dim(1) == 0) {
            throw new IllegalArgumentException("Clustering vectors must have at least one dimension");
        }
    }

    /** Returns the borrowed [sampleCount, dimensions] tensor. */
    public Tensor vectors() {
        return vectors;
    }

    public long sampleCount() {
        return vectors.shape().dim(0);
    }

    public long dimensions() {
        return vectors.shape().dim(1);
    }
}
