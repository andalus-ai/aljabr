package tech.kayys.aljabr.clustering.emeans;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.clustering.api.ClusteringDataset;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class EMeansAlgorithmTest {
    private final ComputeBackend backend = new CpuBackend();
    private final EMeansAlgorithm algorithm = new EMeansAlgorithm();

    @Test
    void clustersSeparatedPointsAndReportsFinalObjective() {
        Tensor vectors = tensor(
                new float[] { 0.0f, 0.0f, 0.1f, 0.0f, 10.0f, 10.0f, 10.1f, 10.0f },
                4, 2);
        try (ClusteringResult result = algorithm.fit(
                new ClusteringDataset(vectors), EMeansConfig.defaults(2, 7L, 8, 2))) {
            assertThat(result.clusterCount()).isEqualTo(2);
            assertThat(result.sampleCount()).isEqualTo(4);
            assertThat(result.clusterOf(0)).isEqualTo(result.clusterOf(1));
            assertThat(result.clusterOf(2)).isEqualTo(result.clusterOf(3));
            assertThat(result.clusterOf(0)).isNotEqualTo(result.clusterOf(2));
            assertThat(result.metrics().objectiveName()).isEqualTo("sum_squared_euclidean");
            assertThat(result.metrics().objectiveValue()).isLessThan(0.02);
            assertThat(result.metadata().algorithmId()).isEqualTo("emeans");
            assertThat(result.metadata().iterations()).isEqualTo(8);
            assertThat(result.centroids().shape().dims()).containsExactly(2, 2);
        } finally {
            vectors.close();
        }
    }

    @Test
    void fixedSeedProducesRepeatableResults() {
        Tensor vectors = tensor(
                new float[] { 0, 0, 1, 0, 9, 9, 10, 9, 5, 5 },
                5, 2);
        try {
            float[] firstCenters;
            int[] firstAssignments;
            try (ClusteringResult first = algorithm.fit(
                    new ClusteringDataset(vectors), EMeansConfig.defaults(2, 99L, 6, 2))) {
                firstCenters = first.centroids().toFloatArray();
                firstAssignments = assignments(first);
            }
            try (ClusteringResult second = algorithm.fit(
                    new ClusteringDataset(vectors), EMeansConfig.defaults(2, 99L, 6, 2))) {
                assertThat(second.centroids().toFloatArray()).containsExactly(firstCenters);
                assertThat(assignments(second)).containsExactly(firstAssignments);
            }
        } finally {
            vectors.close();
        }
    }

    @Test
    void rejectsInvalidClusterCountAndNonFiniteInput() {
        Tensor tooSmall = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        try {
            assertThatThrownBy(() -> algorithm.fit(
                    new ClusteringDataset(tooSmall), EMeansConfig.defaults(3, 1L, 2, 1)))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("must not exceed sample count");
        } finally {
            tooSmall.close();
        }

        Tensor nonFinite = tensor(new float[] { 0, Float.NaN, 1, 2 }, 2, 2);
        try {
            assertThatThrownBy(() -> algorithm.fit(
                    new ClusteringDataset(nonFinite), EMeansConfig.defaults(1, 1L, 2, 1)))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("non-finite");
        } finally {
            nonFinite.close();
        }
    }


    @Test
    void optionallyProducesSlowEmaCentroids() {
        Tensor vectors = tensor(new float[] { 0, 0, 0.2f, 0, 9.8f, 10, 10, 10 }, 4, 2);
        try {
            EMeansConfig enabled = EMeansConfig.defaults(2, 17L, 5, 2)
                    .withSlowEmaStepMultiplier(0.25);
            try (EMeansResult result = algorithm.fit(new ClusteringDataset(vectors), enabled)) {
                assertThat(result.slowEmaCentroids()).isPresent();
                assertThat(result.slowEmaCentroids().orElseThrow().shape().dim(1)).isEqualTo(2);
                assertThat(result.centroids().shape().dim(0)).isEqualTo(2);
            }
            try (EMeansResult result = algorithm.fit(new ClusteringDataset(vectors),
                    EMeansConfig.defaults(2, 17L, 5, 2))) {
                assertThat(result.slowEmaCentroids()).isEmpty();
            }
        } finally { vectors.close(); }
    }

    @Test
    void rejectsInvalidSlowEmaMultiplier() {
        assertThatThrownBy(() -> EMeansConfig.defaults(2, 1L, 2, 2)
                .withSlowEmaStepMultiplier(5.0))
                .isInstanceOf(IllegalArgumentException.class);
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
