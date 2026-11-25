package tech.kayys.aljabr.forecasting.api;

import java.util.Objects;
import java.util.Set;

/**
 * Runtime-discoverable forecasting limits and optional behaviors.
 *
 * <p>These are model capabilities, not backend capabilities. A backend may
 * expose additional device/dtype limits through its own execution contract.
 */
public record ForecastingCapabilities(
        int maximumContextLength,
        int maximumPredictionLength,
        Set<ForecastingFeature> features) {

    public ForecastingCapabilities {
        if (maximumContextLength < 1 || maximumPredictionLength < 1) {
            throw new IllegalArgumentException("forecasting limits must be positive");
        }
        features = Set.copyOf(Objects.requireNonNull(features, "features"));
    }

    public boolean supports(ForecastingFeature feature) {
        return features.contains(Objects.requireNonNull(feature, "feature"));
    }
}
