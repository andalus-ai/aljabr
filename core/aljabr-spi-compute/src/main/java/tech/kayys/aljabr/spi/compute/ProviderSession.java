package tech.kayys.aljabr.spi.compute;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.OperationSupportDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

/**
 * A device-scoped, explicitly owned collection of tensor compute backends.
 *
 * <p>A provider may expose multiple backend instances from one session. Keys
 * are stable provider-defined backend identifiers, not selection policies.
 * Consumers must close the session when execution using it has finished.
 * Closing a session closes each distinct backend instance owned by that
 * session, then releases other session-owned resources. Backends shared across
 * sessions must be managed by the provider instead of being session-owned.
 * Provider closure, in turn, closes all sessions still owned by that provider.</p>
 *
 * <p>Session close is terminal and idempotent. The device descriptor remains
 * available for diagnostics after close, but accessing compute backends after
 * close must fail with {@link IllegalStateException}.</p>
 */
public interface ProviderSession extends AutoCloseable {

    /** Device to which this session is bound. */
    DeviceDescriptor device();

    /**
     * Backends available in this session, keyed by stable backend identifier.
     * The returned map must be non-null and must not be structurally mutable
     * by the caller. Throws {@link IllegalStateException} after this session closes.
     */
    Map<String, ComputeBackend> computeBackends();

    /**
     * Reports how this session can execute an operation under caller-supplied constraints.
     *
     * <p>Operation identifiers and constraint keys are provider-neutral strings.
     * Constraints describe requirements (for example dtype, layout, or shape class);
     * they are not runtime-selection policy. Implementations may override this
     * method when they can make a reliable claim. The default is deliberately
     * {@link OperationSupport#UNKNOWN}; the presence of a method on
     * {@link ComputeBackend} alone is not evidence of support.</p>
     *
     * @param operationId stable, provider-neutral operation identifier
     * @param constraints immutable-by-convention constraint values supplied by the caller
     * @return a non-null support description for the requested operation
     */
    default OperationSupportDescriptor operationSupport(
            String operationId,
            Map<String, String> constraints) {
        Objects.requireNonNull(constraints, "constraints");
        return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
    }

    /**
     * Closes each distinct backend instance in a session-owned backend map.
     * Identity, rather than {@code equals}, defines distinct instances. All
     * backends are attempted even if one close operation fails.
     */
    static void closeBackends(Map<String, ComputeBackend> backends) throws ProviderException {
        Objects.requireNonNull(backends, "backends");
        Set<ComputeBackend> closed = Collections.newSetFromMap(new IdentityHashMap<>());
        ProviderException failure = null;
        for (ComputeBackend backend : backends.values()) {
            Objects.requireNonNull(backend, "backend");
            if (!closed.add(backend)) {
                continue;
            }
            try {
                backend.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) {
                    failure = new ProviderException("Unable to close one or more session-owned compute backends",
                            closeFailure);
                } else {
                    failure.addSuppressed(closeFailure);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Releases resources owned by this session; repeated calls must be harmless. */
    @Override
    void close() throws ProviderException;
}
