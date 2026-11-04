package tech.kayys.aljabr.spi.compute;

import java.util.Map;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

/**
 * A device-scoped, explicitly owned collection of tensor compute backends.
 *
 * <p>A provider may expose multiple backend instances from one session. Keys
 * are stable provider-defined backend identifiers, not selection policies.
 * Consumers must close the session when execution using it has finished.
 * Closing a session releases session-owned resources but does not close the
 * provider itself.
 */
public interface ProviderSession extends AutoCloseable {

    /** Device to which this session is bound. */
    DeviceDescriptor device();

    /**
     * Backends available in this session, keyed by stable backend identifier.
     * The returned map must be non-null and must not be structurally mutable
     * by the caller.
     */
    Map<String, ComputeBackend> computeBackends();

    /** Releases resources owned by this session. */
    @Override
    void close() throws ProviderException;
}