package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

import io.smallrye.mutiny.Uni;

/**
 * Provider interface for model statistics.
 * This allows the registry module to get stats without depending on the engine
 * module.
 * @author bhangun
 */
public interface ModelStatsProvider {

    /**
     * Get model statistics including inference counts.
     * 
     * @param requestId The request/tenant ID
     * @param modelId   The model ID
     * @return Model statistics
     */
    Uni<ModelRegistry.ModelStats> getModelStats(String requestId, String modelId);
}
