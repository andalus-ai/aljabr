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
        assertThat(recommendation.lowDimensionalCaution()).isFalse();
        assertThat(recommendation.largeDatasetAdjustment()).isFalse();
    }

    @Test
    void marksLowDimensionalAndLargeDatasetRecommendations() {
        var recommendation = EMeansParameterScaling.recommend(3_000_000, 4_096, 5, 16);

        assertThat(recommendation.lowDimensionalCaution()).isTrue();
        assertThat(recommendation.largeDatasetAdjustment()).isTrue();
        assertThat(recommendation.batchSize()).isPositive();
        assertThat(recommendation.batchSize()).isLessThanOrEqualTo(3_000_000);
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
