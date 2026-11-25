package tech.kayys.aljabr.forecasting.api;

/**
 * Provider-neutral forecasting inference contract.
 *
 * <p>Implementations declare model-specific limits and supported input
 * features separately; this interface deliberately contains no tensor,
 * checkpoint-format, device, or model-family assumptions.
 */
public interface ForecastingModel extends AutoCloseable {
    ForecastingCapabilities capabilities();

    ForecastResult forecast(ForecastRequest request);

    default String modelId() {
        return getClass().getSimpleName();
    }

    @Override
    default void close() {
        // Stateless implementations need not override.
    }
}
