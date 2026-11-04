package tech.kayys.aljabr.spi.compute;

import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

/**
 * Optional compute-facing extension for providers that expose tensor backends.
 *
 * <p>The foundational {@link AljabrProvider} contract remains independent of the
 * tensor module. Implementations opt into this extension when they can open
 * device-scoped sessions that expose {@code ComputeBackend} instances.
 */
public interface ComputeProvider extends AljabrProvider {

    /**
     * Opens a session for a device previously reported by this provider.
     *
     * <p>The provider must reject a descriptor whose provider id does not match
     * {@code descriptor().id()}. Session creation does not imply that the
     * consumer has permitted provider-level fallback.
     */
    ProviderSession openSession(DeviceDescriptor device) throws ProviderException;
}