# Aljabr Time Series API

Provider-neutral contracts for timestamped observations and covariates.

- `TimeSeries`: immutable numeric target observations with missing-value masks.
- `CovariateSeries`: real or categorical covariates, including known-future features.
- `TimeSeriesGroup`: related targets and covariates that may share information during forecasting.

The API does not infer frequency, timezone, calendar semantics, resampling, or imputation. Those belong in explicit preprocessing components so models cannot silently reinterpret source data.

`InstanceNormalizer` provides reusable per-series population-mean / population-standard-deviation normalization with explicit missing masks, configurable epsilon, optional arcsinh compression, and an immutable inverse-transform state. `TimeSeriesPatcher` provides left-padded sliding-window extraction with explicit patch masks and configurable stride. Model families select the settings they require; the shared API does not impose one normalization or patching policy.
