package tech.kayys.aljabr.spi.spec;

import tech.kayys.aljabr.core.tensor.DeviceType;

/**
 * Represents the resource requirements for running a model.
 * @author bhangun
 */
public record ResourceRequirements(
                MemoryRequirements memory,
                ComputeRequirements compute,
                StorageRequirements storage) {
}