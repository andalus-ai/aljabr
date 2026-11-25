package tech.kayys.aljabr.forecasting.api;

/** Optional forecasting behaviors exposed by a model implementation. */
public enum ForecastingFeature {
    UNIVARIATE,
    MULTIVARIATE,
    CROSS_SERIES_LEARNING,
    PAST_REAL_COVARIATES,
    PAST_CATEGORICAL_COVARIATES,
    FUTURE_REAL_COVARIATES,
    FUTURE_CATEGORICAL_COVARIATES,
    PROBABILISTIC_QUANTILES
}
