package tech.kayys.aljabr.clustering.emeans;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class EMeansParameterScalingTest {
    @Test
    void recommendsPowerOfTwoBatchAndBoundedStep() {
        var recommendation = EMeansParameterScaling.recommend(1_000_000, 1_024, 4, 768);

        assertThat(recommendation.batchSize()).isPositive();
        assertThat(Integer.bitCount(recommendation.batchSize())).isEqualTo(1);
        assertThat(recommendation.batchSize()).isLessThanOrEqualTo(1_000_000);
        assertThat(recommendation.initialStepSize()).isPositive().isLessThanOrEqualTo(1.0);
        assertThat(recommendation.largeDatasetAdjustment()).isFalse();
    }

    @Test
    void marksLowDimensionalAndLargeDatasetRecommendations() {
        var recommendation = EMeansParameterScaling.recommend(3_000_000, 4_096, 5, 16);

        assertThat(recommendation.largeDatasetAdjustment()).isTrue();
        assertThat(recommendation.batchSize()).isPositive();
        assertThat(recommendation.batchSize()).isLessThanOrEqualTo(3_000_000);
    }

    @Test
    void automaticConfigUsesTheRecommendation() {
        var recommendation = EMeansParameterScaling.recommend(100_000, 256, 4, 128);
        var config = EMeansConfig.auto(256, 42L, 4, 100_000, 128);

        assertThat(config.batchSize()).isEqualTo(recommendation.batchSize());
        assertThat(config.initialStepSize()).isEqualTo(recommendation.initialStepSize());
        assertThat(config.reviveDeadClusters()).isTrue();
    }

    @Test
    void rejectsInvalidProblemDimensions() {
        assertThatThrownBy(() -> EMeansParameterScaling.recommend(0, 1, 1, 8))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> EMeansParameterScaling.recommend(10, 11, 1, 8))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> EMeansParameterScaling.recommend(10, 2, 0, 8))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
