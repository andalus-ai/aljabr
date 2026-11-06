package tech.kayys.aljabr.backend.metal;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.metal.binding.MetalBinding;
import tech.kayys.aljabr.spi.compute.ComputeProvider;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.CapabilityDescriptor;
import tech.kayys.aljabr.spi.provider.CapabilityStability;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

/**
 * Device-scoped provider adapter for the existing Metal compute backend.
 * Discovery is platform-only and never loads the native Metal bridge.
 */
public final class MetalComputeProvider implements ComputeProvider {
    public static final String PROVIDER_ID = "andalus/metal";
    public static final String BACKEND_ID = "metal-default";

    private static final ProviderDescriptor DESCRIPTOR = new ProviderDescriptor(
            PROVIDER_ID, "0.1.0", ">=0.1.0 <1.0.0",
            Set.of(new CapabilityDescriptor("tensor.compute", "1", CapabilityStability.STABLE)));

    private final BooleanSupplier platformAvailable;
    private final Supplier<? extends ComputeBackend> backendFactory;
    private final DeviceDescriptor device = new DeviceDescriptor(
            PROVIDER_ID, "metal:0", "Apple Metal GPU", DeviceKind.GPU,
            null, null, "Apple", System.getProperty("os.arch", "unknown"), null, null);

    private final Set<MetalProviderSession> sessions = ConcurrentHashMap.newKeySet();

    private boolean initialized;
    private boolean closed;

    /** Creates the production adapter without initializing native Metal. */
    public MetalComputeProvider() {
        this(MetalComputeProvider::isMacOs, MetalComputeProvider::createNativeBackend);
    }

    // Package-private seams keep provider lifecycle tests independent of Metal hardware.
    MetalComputeProvider(BooleanSupplier platformAvailable,
                         Supplier<? extends ComputeBackend> backendFactory) {
        this.platformAvailable = Objects.requireNonNull(platformAvailable, "platformAvailable");
        this.backendFactory = Objects.requireNonNull(backendFactory, "backendFactory");
    }

    @Override
    public ProviderDescriptor descriptor() {
        return DESCRIPTOR;
    }

    @Override
    public synchronized void initialize(ProviderContext context) throws ProviderException {
        Objects.requireNonNull(context, "context");
        if (closed) {
            throw new ProviderException("Metal provider is closed");
        }
        if (initialized) {
            throw new ProviderException("Metal provider is already initialized");
        }
        // Native loading is deliberately deferred until openSession().
        initialized = true;
    }

    @Override
    public synchronized List<DeviceDescriptor> devices() throws ProviderException {
        if (closed) {
            throw new ProviderException("Metal provider is closed");
        }
        return platformAvailable.getAsBoolean() ? List.of(device) : List.of();
    }

    @Override
    public synchronized ProviderSession openSession(DeviceDescriptor requested) throws ProviderException {
        if (closed) {
            throw new ProviderException("Metal provider is closed");
        }
        if (!initialized) {
            throw new ProviderException("Metal provider has not been initialized");
        }
        if (!platformAvailable.getAsBoolean()) {
            throw new ProviderException("Metal is only available on macOS hosts");
        }
        if (!device.equals(Objects.requireNonNull(requested, "device"))) {
            throw new ProviderException("Device does not belong to provider " + PROVIDER_ID + ": "
                    + requested.deviceId());
        }

        try {
            ComputeBackend backend = Objects.requireNonNull(backendFactory.get(), "backendFactory result");
            MetalProviderSession session = new MetalProviderSession(this, device, backend);
            sessions.add(session);
            return session;
        } catch (RuntimeException e) {
            throw new ProviderException("Unable to open a native Metal compute session", e);
        }
    }

    @Override
    public void close() {
        List<MetalProviderSession> openSessions;
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            initialized = false;
            openSessions = new ArrayList<>(sessions);
            sessions.clear();
        }
        openSessions.forEach(MetalProviderSession::close);
    }

    private static boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase().contains("mac");
    }

    private static ComputeBackend createNativeBackend() {
        // MetalComputeBackend currently initializes the shared FFM binding and
        // contains a legacy CPU fallback. Refuse to publish a Metal session if
        // that fallback was activated: provider selection must not silently
        // turn a requested Metal session into CPU execution.
        MetalComputeBackend backend = new MetalComputeBackend();
        if (!MetalBinding.getInstance().isRuntimeActive()) {
            throw new IllegalStateException("Metal native runtime is not active; refusing CPU fallback session");
        }
        return backend;
    }

    private static final class MetalProviderSession implements ProviderSession {
        private final MetalComputeProvider owner;
        private final DeviceDescriptor device;
        private final Map<String, ComputeBackend> backends;
        private boolean closed;

        private MetalProviderSession(MetalComputeProvider owner, DeviceDescriptor device, ComputeBackend backend) {
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
            if (closed) {
                throw new IllegalStateException("Metal provider session is closed");
            }
            return backends;
        }

        @Override
        public synchronized void close() {
            if (closed) {
                return;
            }
            // MetalBinding is currently a process-scoped singleton; the legacy
            // backend exposes no per-session native release operation.
            closed = true;
            owner.sessions.remove(this);
        }
    }
}