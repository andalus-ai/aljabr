package tech.kayys.aljabr.backend.hat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
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

    /**
     * Shape-only operations that retain the input dtype and do not interpret
     * tensor elements as floats. These can be delegated without a dtype hint.
     */
    private static final Set<String> CPU_DTYPE_PRESERVING_OPERATIONS = Set.of(
            "tensor.reshape", "tensor.flatten", "tensor.unsqueeze", "tensor.squeeze");

    private static final Set<String> CPU_DELEGATED_OPERATIONS = Set.of(
            "tensor.add", "tensor.sub", "tensor.mul", "tensor.div",
            "tensor.mul_scalar", "tensor.div_scalar", "tensor.add_scalar",
            "tensor.reshape", "tensor.slice", "tensor.split",
            "tensor.flatten", "tensor.unsqueeze", "tensor.squeeze", "tensor.transpose",
            "tensor.softmax", "tensor.log_softmax", "tensor.mean", "tensor.sum",
            "tensor.max", "tensor.abs", "tensor.relu", "tensor.gelu", "tensor.silu",
            "tensor.sigmoid", "tensor.tanh", "tensor.sqrt", "tensor.exp", "tensor.log",
            "tensor.pow", "tensor.layer_norm", "tensor.rms_norm", "tensor.batch_norm",
            "tensor.conv2d", "tensor.max_pool2d", "tensor.adaptive_avg_pool2d",
            "tensor.dropout", "tensor.cross_entropy", "tensor.binary_cross_entropy",
            "tensor.zeros_like");

    private static final Set<String> UNSUPPORTED_OPERATIONS = Set.of(
            "tensor.cat", "tensor.upsample_nearest2d", "tensor.attention",
            "tensor.attention.masked", "tensor.rope");

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

        CpuBackend cpuFallback = new CpuBackend();
        HatComputeBackend hatBackend = new HatComputeBackend(cpuFallback);
        HatProviderSession session = new HatProviderSession(this, device, hatBackend, cpuFallback);
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

        private HatProviderSession(
                HatComputeProvider owner,
                DeviceDescriptor device,
                HatComputeBackend hatBackend,
                CpuBackend cpuFallback) {
            this.owner = owner;
            this.device = device;
            this.backends = Map.of(
                    BACKEND_ID, hatBackend,
                    CPU_FALLBACK_BACKEND_ID, cpuFallback);
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

            if (CPU_DTYPE_PRESERVING_OPERATIONS.contains(operationId)) {
                return new OperationSupportDescriptor(
                        operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }

            if ("tensor.layer_norm".equals(operationId)) {
                return normalizationSupport(operationId, constraints, false, false, true);
            }

            if ("tensor.rms_norm".equals(operationId)) {
                return normalizationSupport(operationId, constraints, true, false, false);
            }

            if ("tensor.batch_norm".equals(operationId)) {
                String training = constraints.get("training");
                if (training == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!"true".equalsIgnoreCase(training) && !"false".equalsIgnoreCase(training)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (Boolean.parseBoolean(training)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                String rank = constraints.get("rank");
                if (rank == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                try {
                    if (Integer.parseInt(rank) < 2) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                } catch (NumberFormatException invalidRank) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                String runningMeanDtype = constraints.get("running_mean_dtype");
                String runningVarDtype = constraints.get("running_var_dtype");
                if (runningMeanDtype == null || runningVarDtype == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!isF32(runningMeanDtype) || !isF32(runningVarDtype)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                return normalizationSupport(operationId, constraints, false, false, true);
            }

            if ("tensor.cross_entropy".equals(operationId)) {
                String predictionDtype = constraints.get("dtype");
                String targetDtype = constraints.get("target_dtype");
                if (predictionDtype == null || targetDtype == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!"F32".equalsIgnoreCase(predictionDtype)
                        || !"I32".equalsIgnoreCase(targetDtype)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                return new OperationSupportDescriptor(
                        operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }

            if ("tensor.binary_cross_entropy".equals(operationId)) {
                String predictionDtype = constraints.get("dtype");
                String targetDtype = constraints.get("target_dtype");
                if (predictionDtype == null || targetDtype == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!"F32".equalsIgnoreCase(predictionDtype)
                        || !"F32".equalsIgnoreCase(targetDtype)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                return new OperationSupportDescriptor(
                        operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }

            if (CPU_DELEGATED_OPERATIONS.contains(operationId)) {
                String dtype = constraints.get("dtype");
                if (dtype == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!"F32".equalsIgnoreCase(dtype)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                return new OperationSupportDescriptor(
                        operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }
            if (UNSUPPORTED_OPERATIONS.contains(operationId)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
        }

        /**
         * Models the CPU normalization kernels' float-only memory layouts.
         * Optional affine tensors are explicit: callers provide *_present,
         * and a dtype whenever the corresponding tensor is present.
         */
        private OperationSupportDescriptor normalizationSupport(
                String operationId,
                Map<String, String> constraints,
                boolean weightRequired,
                boolean biasRequired,
                boolean checkBias) {
            String inputDtype = constraints.get("dtype");
            if (inputDtype == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(inputDtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }

            OperationSupport weightSupport = optionalNormalizationTensorSupport(
                    constraints, "weight", weightRequired);
            OperationSupport biasSupport = checkBias
                    ? optionalNormalizationTensorSupport(constraints, "bias", biasRequired)
                    : OperationSupport.DELEGATED;
            if (weightSupport == OperationSupport.UNSUPPORTED
                    || biasSupport == OperationSupport.UNSUPPORTED) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            if (weightSupport == OperationSupport.UNKNOWN || biasSupport == OperationSupport.UNKNOWN) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private OperationSupport optionalNormalizationTensorSupport(
                Map<String, String> constraints, String tensorName, boolean required) {
            String present = constraints.get(tensorName + "_present");
            if (present == null) {
                if (!required) return OperationSupport.UNKNOWN;
                String dtype = constraints.get(tensorName + "_dtype");
                if (dtype == null) return OperationSupport.UNKNOWN;
                return isF32(dtype) ? OperationSupport.DELEGATED : OperationSupport.UNSUPPORTED;
            }
            if (!"true".equalsIgnoreCase(present) && !"false".equalsIgnoreCase(present)) {
                return OperationSupport.UNKNOWN;
            }
            if ("false".equalsIgnoreCase(present)) {
                return required ? OperationSupport.UNSUPPORTED : OperationSupport.DELEGATED;
            }
            String dtype = constraints.get(tensorName + "_dtype");
            if (dtype == null) return OperationSupport.UNKNOWN;
            return isF32(dtype) ? OperationSupport.DELEGATED : OperationSupport.UNSUPPORTED;
        }

        private static boolean isF32(String dtype) {
            return "F32".equalsIgnoreCase(dtype);
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
