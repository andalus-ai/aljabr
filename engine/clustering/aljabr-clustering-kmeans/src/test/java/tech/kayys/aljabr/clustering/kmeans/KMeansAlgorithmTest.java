package tech.kayys.aljabr.clustering.kmeans;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.clustering.api.ClusteringDataset;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.clustering.api.DistanceMetric;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class KMeansAlgorithmTest {
    private final ComputeBackend backend = new CpuBackend();
    private final KMeansAlgorithm algorithm = new KMeansAlgorithm();

    @Test
    void clustersSeparatedPointsAndReportsObjective() {
        Tensor vectors = tensor(
                new float[] { 0.0f, 0.0f, 0.1f, 0.0f, 10.0f, 10.0f, 10.1f, 10.0f },
                4, 2);
        try (ClusteringResult result = algorithm.fit(
                new ClusteringDataset(vectors), KMeansConfig.defaults(2, 17L))) {
            assertThat(result.clusterCount()).isEqualTo(2);
            assertThat(result.sampleCount()).isEqualTo(4);
            assertThat(result.clusterOf(0)).isEqualTo(result.clusterOf(1));
            assertThat(result.clusterOf(2)).isEqualTo(result.clusterOf(3));
            assertThat(result.clusterOf(0)).isNotEqualTo(result.clusterOf(2));
            assertThat(result.metrics().objectiveName()).isEqualTo("sum_squared_euclidean");
            assertThat(result.metrics().objectiveValue()).isLessThan(0.02);
            assertThat(result.metadata().algorithmId()).isEqualTo("kmeans");
            assertThat(result.metadata().iterations()).isGreaterThan(0);
            assertThat(result.centroids().shape().dims()).containsExactly(2, 2);
        } finally {
            vectors.close();
        }
    }

    @Test
    void fixedSeedProducesRepeatableAssignmentsAndCentroids() {
        Tensor vectors = tensor(
                new float[] { 0, 0, 1, 0, 9, 9, 10, 9, 5, 5 },
                5, 2);
        try {
            float[] firstCentroids;
            int[] firstAssignments;
            try (ClusteringResult first = algorithm.fit(
                    new ClusteringDataset(vectors), KMeansConfig.defaults(2, 1234L))) {
                firstCentroids = first.centroids().toFloatArray();
                firstAssignments = assignments(first);
            }
            try (ClusteringResult second = algorithm.fit(
                    new ClusteringDataset(vectors), KMeansConfig.defaults(2, 1234L))) {
                assertThat(second.centroids().toFloatArray()).containsExactly(firstCentroids);
                assertThat(assignments(second)).containsExactly(firstAssignments);
            }
        } finally {
            vectors.close();
        }
    }

    @Test
    void cosineMetricGroupsVectorsByDirection() {
        Tensor vectors = tensor(
                new float[] { 1, 0, 2, 0, 0, 1, 0, 3 },
                4, 2);
        KMeansConfig config = new KMeansConfig(
                2, 9L, DistanceMetric.COSINE, 100, 1.0e-5,
                KMeansInitialization.KMEANS_PLUS_PLUS);
        try {
            try (ClusteringResult result = algorithm.fit(new ClusteringDataset(vectors), config)) {
                assertThat(result.clusterOf(0)).isEqualTo(result.clusterOf(1));
                assertThat(result.clusterOf(2)).isEqualTo(result.clusterOf(3));
                assertThat(result.clusterOf(0)).isNotEqualTo(result.clusterOf(2));
                assertThat(result.metrics().objectiveName()).isEqualTo("sum_cosine_distance");
            }
        } finally {
            vectors.close();
        }
    }

    @Test
    void rejectsTooManyClustersAndOutOfRangeAssignmentLookups() {
        Tensor vectors = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        try {
            assertThatThrownBy(() -> algorithm.fit(
                    new ClusteringDataset(vectors), KMeansConfig.defaults(3, 1L)))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("must not exceed sample count");

            try (ClusteringResult result = algorithm.fit(
                    new ClusteringDataset(vectors), KMeansConfig.defaults(1, 1L))) {
                assertThatThrownBy(() -> result.clusterOf(-1))
                        .isInstanceOf(IndexOutOfBoundsException.class);
                assertThatThrownBy(() -> result.clusterOf(2))
                        .isInstanceOf(IndexOutOfBoundsException.class);
            }
        } finally {
            vectors.close();
        }
    }

    @Test
    void rejectsNonFiniteInputValues() {
        Tensor vectors = tensor(new float[] { 0, Float.NaN, 1, 2 }, 2, 2);
        try {
            assertThatThrownBy(() -> algorithm.fit(
                    new ClusteringDataset(vectors), KMeansConfig.defaults(1, 1L)))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("non-finite");
        } finally {
            vectors.close();
        }
    }

    private Tensor tensor(float[] values, long... shape) {
        return TensorFactory.of(values, backend, shape);
    }

    private static int[] assignments(ClusteringResult result) {
        int[] values = new int[Math.toIntExact(result.sampleCount())];
        for (int i = 0; i < values.length; i++) {
            values[i] = result.clusterOf(i);
        }
        return values;
    }
}
