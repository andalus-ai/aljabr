package tech.kayys.aljabr.backend.cpu;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.aljabr.spi.compute.ComputeProvider;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.CapabilityDescriptor;
import tech.kayys.aljabr.spi.provider.CapabilityStability;
import tech.kayys.aljabr.spi.provider.DeviceDescriptor;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

/** Pure-Java CPU implementation of the compute provider SPI. */
public final class CpuComputeProvider implements ComputeProvider {
    public static final String PROVIDER_ID = "andalus/cpu";
    public static final String BACKEND_ID = "cpu-default";

    private static final ProviderDescriptor DESCRIPTOR = new ProviderDescriptor(
            PROVIDER_ID, "0.1.0", "1",
            Set.of(new CapabilityDescriptor("tensor.compute", "1", CapabilityStability.STABLE)));

    private final DeviceDescriptor device = new DeviceDescriptor(
            PROVIDER_ID, "cpu:0", "Host CPU", DeviceKind.CPU, null, null,
            "generic", System.getProperty("os.arch", "unknown"), null,
            System.getProperty("java.vm.name", "unknown"));

    private boolean initialized;
    private boolean closed;

    @Override
    public ProviderDescriptor descriptor() {
        return DESCRIPTOR;
    }

    @Override
    public synchronized void initialize(ProviderContext context) throws ProviderException {
        Objects.requireNonNull(context, "context");
        if (closed) throw new ProviderException("CPU provider is closed");
        initialized = true;
    }

    @Override
    public List<DeviceDescriptor> devices() throws ProviderException {
        synchronized (this) {
            if (closed) throw new ProviderException("CPU provider is closed");
        }
        return List.of(device);
    }

    @Override
    public synchronized ProviderSession openSession(DeviceDescriptor requested) throws ProviderException {
        if (closed) throw new ProviderException("CPU provider is closed");
        if (!initialized) throw new ProviderException("CPU provider has not been initialized");
        if (!device.equals(Objects.requireNonNull(requested, "device"))) {
            throw new ProviderException("Device does not belong to provider " + PROVIDER_ID + ": " + requested.deviceId());
        }
        return new CpuProviderSession(device, new CpuBackend());
    }

    @Override
    public synchronized void close() {
        closed = true;
        initialized = false;
    }

    private static final class CpuProviderSession implements ProviderSession {
        private final DeviceDescriptor device;
        private final Map<String, tech.kayys.aljabr.core.backend.ComputeBackend> backends;
        private boolean closed;

        private CpuProviderSession(DeviceDescriptor device, CpuBackend backend) {
            this.device = device;
            this.backends = Map.of(BACKEND_ID, backend);
        }

        @Override
        public DeviceDescriptor device() {
            return device;
        }

        @Override
        public synchronized Map<String, tech.kayys.aljabr.core.backend.ComputeBackend> computeBackends() {
            if (closed) throw new IllegalStateException("CPU provider session is closed");
            return backends;
        }

        @Override
        public synchronized void close() {
            closed = true;
        }
    }
}