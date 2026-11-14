package tech.kayys.aljabr.backend.hat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.spi.compute.ComputeProvider;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.CapabilityDescriptor;
import tech.kayys.aljabr.spi.provider.CapabilityStability;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.OperationSupportDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

/**
 * ServiceLoader provider for the optional Project Babylon HAT Java backend.
 *
 * <p>The provider reports a CPU execution device because the currently wired
 * HAT Java backends execute on the host CPU. Discovery does not construct an
 * Accelerator; Babylon runtime initialization is deferred until a HAT kernel
 * is first dispatched.</p>
 */
public final class HatComputeProvider implements ComputeProvider {
    public static final String PROVIDER_ID = "andalus/hat-java";
    public static final String BACKEND_ID = "hat-java-default";
    public static final String CPU_FALLBACK_BACKEND_ID = "cpu-fallback";

    private static final ProviderDescriptor DESCRIPTOR = new ProviderDescriptor(
            PROVIDER_ID, "0.1.0", ">=0.1.0 <1.0.0",
            Set.of(new CapabilityDescriptor("tensor.compute", "1", CapabilityStability.EXPERIMENTAL)));

    private final DeviceDescriptor device = new DeviceDescriptor(
            PROVIDER_ID, "hat-java:0", "Project Babylon HAT Java runtime", DeviceKind.CPU,
            null, null, "Oracle", System.getProperty("os.arch", "unknown"), null,
            System.getProperty("java.runtime.version", "unknown"));
    private final Set<HatProviderSession> sessions = ConcurrentHashMap.newKeySet();

    private boolean initialized;
    private boolean closed;

    @Override
    public ProviderDescriptor descriptor() {
        return DESCRIPTOR;
    }

    @Override
    public synchronized void initialize(ProviderContext context) throws ProviderException {
        Objects.requireNonNull(context, "context");
        if (closed) throw new ProviderException("HAT provider is closed");
        if (initialized) throw new ProviderException("HAT provider is already initialized");
        initialized = true;
    }

    @Override
    public synchronized List<DeviceDescriptor> devices() throws ProviderException {
        if (closed) throw new ProviderException("HAT provider is closed");
        return List.of(device);
    }

    @Override
    public synchronized ProviderSession openSession(DeviceDescriptor requested) throws ProviderException {
        if (closed) throw new ProviderException("HAT provider is closed");
        if (!initialized) throw new ProviderException("HAT provider has not been initialized");
        if (!device.equals(Objects.requireNonNull(requested, "device"))) {
            throw new ProviderException("Device does not belong to provider " + PROVIDER_ID + ": "
                    + requested.deviceId());
        }

        HatProviderSession session = new HatProviderSession(this, device, new HatComputeBackend());
        sessions.add(session);
        return session;
    }

    @Override
    public void close() {
        List<HatProviderSession> openSessions;
        synchronized (this) {
            if (closed) return;
            closed = true;
            initialized = false;
            openSessions = new ArrayList<>(sessions);
            sessions.clear();
        }

        RuntimeException failure = null;
        for (HatProviderSession session : openSessions) {
            try {
                session.close();
            } catch (ProviderException closeFailure) {
                if (failure == null) {
                    failure = new IllegalStateException("Unable to close all HAT provider sessions", closeFailure);
                } else {
                    failure.addSuppressed(closeFailure);
                }
            }
        }
        if (failure != null) throw failure;
    }

    private static final class HatProviderSession implements ProviderSession {
        private final HatComputeProvider owner;
        private final DeviceDescriptor device;
        private final Map<String, ComputeBackend> backends;
        private boolean closed;

        private HatProviderSession(HatComputeProvider owner, DeviceDescriptor device, HatComputeBackend backend) {
            this.owner = owner;
            this.device = device;
            this.backends = Map.of(BACKEND_ID, backend);
        }

        @Override
        public DeviceDescriptor device() {
            return device;
        }

        @Override
        public synchronized Map<String, ComputeBackend> computeBackends() {
            if (closed) throw new IllegalStateException("HAT provider session is closed");
            return backends;
        }

        @Override
        public synchronized OperationSupportDescriptor operationSupport(
                String operationId, Map<String, String> constraints) {
            Objects.requireNonNull(operationId, "operationId");
            Objects.requireNonNull(constraints, "constraints");
            if (closed) throw new IllegalStateException("HAT provider session is closed");

            if ("tensor.matmul".equals(operationId)) {
                String dtype = constraints.get("dtype");
                String rank = constraints.get("rank");
                if (dtype == null || rank == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if ("F32".equalsIgnoreCase(dtype) && "2".equals(rank)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.NATIVE, null);
                }
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }

            // HatComputeBackend delegates every operation without a HAT kernel
            // to its CPU fallback. Keep that distinction visible to selectors.
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        @Override
        public synchronized void close() throws ProviderException {
            if (closed) return;
            closed = true;
            owner.sessions.remove(this);
            ProviderSession.closeBackends(backends);
        }
    }
}
