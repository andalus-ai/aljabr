package tech.kayys.aljabr.clustering.emeans;

import java.util.Optional;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.core.tensor.Tensor;

/** E-means result with an optional out-of-sample-oriented centroid set. */
public interface EMeansResult extends ClusteringResult {
    /** Optional slow-EMA centroids, owned by this result and valid until close. */
    Optional<Tensor> slowEmaCentroids();
}
