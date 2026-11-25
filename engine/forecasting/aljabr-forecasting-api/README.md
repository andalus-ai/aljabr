# Aljabr Forecasting API

Provider-neutral forecasting request, result, model, and capability contracts.

Forecast requests carry explicit timestamps and ordered quantile levels. Results preserve target IDs and return one forecast array per requested quantile. `ForecastingCapabilities` describes model limits and optional behaviors independently of execution backend capabilities.

No specific model format, tensor implementation, device, or runtime is required by these contracts.
