package tech.kayys.aljabr.spi.spec;

import tech.kayys.aljabr.core.tensor.DeviceType;

/**
 * Represents a device that is supported by a model, including minimum
 * requirements.
 * @author bhangun
 */
public record SupportedDevice(
        DeviceType type,
        String minVersion,
        MemoryRequirements memoryRequirements) {

    public SupportedDevice {
        if (type == null) {
            throw new IllegalArgumentException("Device type cannot be null");
        }
    }
}