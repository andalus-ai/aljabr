package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

/**
 * Resource utilization metrics for a model runner
 * @author bhangun
 */
public record ResourceMetrics(
                long cpuUsagePercent,
                long memoryUsageBytes,
                long gpuUsagePercent,
                long vramUsageBytes,
                int activeRequests) {
}
