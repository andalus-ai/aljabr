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
            "tensor.max", "tensor.sign", "tensor.abs", "tensor.relu", "tensor.gelu", "tensor.silu",
            "tensor.sigmoid", "tensor.tanh", "tensor.sqrt", "tensor.exp", "tensor.log",
            "tensor.pow", "tensor.layer_norm", "tensor.rms_norm", "tensor.batch_norm",
            "tensor.conv2d", "tensor.max_pool2d", "tensor.adaptive_avg_pool2d",
            "tensor.dropout", "tensor.cross_entropy", "tensor.binary_cross_entropy",
            "tensor.zeros_like");

    private static final Set<String> UNSUPPORTED_OPERATIONS = Set.of();

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

            if ("tensor.attention".equals(operationId)) {
                return attentionSupport(operationId, constraints, false);
            }
            if ("tensor.attention.masked".equals(operationId)) {
                return attentionSupport(operationId, constraints, true);
            }
            if ("tensor.attention.gqa".equals(operationId)) {
                return gqaSupport(operationId, constraints);
            }
            if ("tensor.upsample_nearest2d".equals(operationId)) {
                return upsampleNearest2dSupport(operationId, constraints);
            }
            if ("tensor.cat".equals(operationId)) {
                return catSupport(operationId, constraints);
            }

            /*
             * Capability metadata contract for the two matmul operands:
             * - dtype / rank describe the left operand.
             * - rhs_dtype / rhs_rank describe the right operand.
             *
             * The caller must populate metadata from both operands. Do not
             * infer the right operand from the left: mixed-dtype or non-matrix
             * inputs must never be advertised as native HAT matmul. Missing
             * metadata remains UNKNOWN so the planner can choose a safe path.
             */
            if ("tensor.matmul".equals(operationId)) {
                String leftDtype = constraints.get("dtype");
                String rightDtype = constraints.get("rhs_dtype");
                String leftRankValue = constraints.get("rank");
                String rightRankValue = constraints.get("rhs_rank");
                if (leftDtype == null || rightDtype == null
                        || leftRankValue == null || rightRankValue == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!isF32(leftDtype) || !isF32(rightDtype)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                Integer leftRank = parseInteger(leftRankValue);
                Integer rightRank = parseInteger(rightRankValue);
                if (leftRank == null || rightRank == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (leftRank != 2 || rightRank != 2) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                return new OperationSupportDescriptor(operationId, OperationSupport.NATIVE, null);
            }

            if (CPU_DTYPE_PRESERVING_OPERATIONS.contains(operationId)) {
                return contiguousShapeOperationSupport(operationId, constraints);
            }

            if ("tensor.rope".equals(operationId)) {
                return ropeSupport(operationId, constraints);
            }
            if ("tensor.embedding".equals(operationId)) {
                return embeddingSupport(operationId, constraints);
            }

            if ("tensor.pow".equals(operationId)) {
                String dtype = constraints.get("dtype");
                if (dtype == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                if (!isF32(dtype)) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }
            if ("tensor.slice".equals(operationId)) return sliceSupport(operationId, constraints);
            if ("tensor.transpose".equals(operationId)) return transposeSupport(operationId, constraints);
            if ("tensor.split".equals(operationId)) return splitSupport(operationId, constraints);

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

            if ("tensor.conv2d".equals(operationId)) return convolutionSupport(operationId, constraints);
            if ("tensor.max_pool2d".equals(operationId))
                return poolingSupport(operationId, constraints, "kernel_size", "stride", "padding");
            if ("tensor.adaptive_avg_pool2d".equals(operationId))
                return poolingSupport(operationId, constraints, "output_h", "output_w", null);

            if ("tensor.softmax".equals(operationId) || "tensor.log_softmax".equals(operationId)) {
                return softmaxSupport(operationId, constraints);
            }

            if ("tensor.mean".equals(operationId)
                    || "tensor.sum".equals(operationId)
                    || "tensor.max".equals(operationId)) {
                return reductionSupport(operationId, constraints);
            }

            if ("tensor.dropout".equals(operationId)) {
                return dropoutSupport(operationId, constraints);
            }

            if ("tensor.zeros_like".equals(operationId)) {
                String dtype = constraints.get("dtype");
                if (dtype == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                if (!isF32(dtype)) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }

            if (Set.of("tensor.add", "tensor.sub", "tensor.mul", "tensor.div").contains(operationId)) {
                String leftDtype = constraints.get("dtype");
                String rightDtype = constraints.get("rhs_dtype");
                if (leftDtype == null || rightDtype == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (!isF32(leftDtype) || !isF32(rightDtype)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                return new OperationSupportDescriptor(
                        operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
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
         * CPU reshape and its shape-only wrappers preserve dtype, but currently
         * require a contiguous input layout. Missing layout metadata must not
         * cause the planner to assume that a non-contiguous view is executable.
         */
        private OperationSupportDescriptor contiguousShapeOperationSupport(
                String operationId, Map<String, String> constraints) {
            String contiguous = constraints.get("contiguous");
            if (contiguous == null
                    || (!"true".equalsIgnoreCase(contiguous)
                    && !"false".equalsIgnoreCase(contiguous))) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!Boolean.parseBoolean(contiguous)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * CpuBackend attention fallback supports matching F32 Q/K/V ranks:
         * rank-2 for unmasked, non-causal attention and rank-4 for the
         * batched/head-aware path. Shape compatibility is validated by the
         * CPU execution method because the provider metadata does not carry
         * every dimension.
         */
        private OperationSupportDescriptor attentionSupport(
                String operationId, Map<String, String> constraints, boolean maskedOperation) {
            String queryDtype = constraints.get("dtype");
            String keyDtype = constraints.get("key_dtype");
            String valueDtype = constraints.get("value_dtype");
            String queryRankValue = constraints.get("rank");
            String keyRankValue = constraints.get("key_rank");
            String valueRankValue = constraints.get("value_rank");
            String maskDtype = constraints.get("mask_dtype");
            String maskRankValue = constraints.get("mask_rank");
            String causalValue = constraints.get("causal");
            if (queryDtype == null || keyDtype == null || valueDtype == null
                    || queryRankValue == null || keyRankValue == null || valueRankValue == null
                    || (maskedOperation && causalValue == null)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(queryDtype) || !isF32(keyDtype) || !isF32(valueDtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            Integer queryRank = parseInteger(queryRankValue);
            Integer keyRank = parseInteger(keyRankValue);
            Integer valueRank = parseInteger(valueRankValue);
            if (queryRank == null || keyRank == null || valueRank == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!queryRank.equals(keyRank) || !queryRank.equals(valueRank)
                    || (queryRank != 2 && queryRank != 4)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            boolean causal;
            if (causalValue == null) {
                causal = false;
            } else if ("true".equalsIgnoreCase(causalValue) || "false".equalsIgnoreCase(causalValue)) {
                causal = Boolean.parseBoolean(causalValue);
            } else {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (maskedOperation && queryRank != 4) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            if (maskedOperation) {
                if ((maskDtype == null) != (maskRankValue == null)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (maskDtype != null) {
                    if (!isF32(maskDtype)) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                    Integer maskRank = parseInteger(maskRankValue);
                    if (maskRank == null) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                    }
                    if (maskRank != 2) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                }
            } else if (maskDtype != null || maskRankValue != null || causal) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * CpuBackend nearest-neighbor upsampling accepts NCHW F32 input with
         * positive input and output spatial dimensions.
         */
        private OperationSupportDescriptor upsampleNearest2dSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            String rankValue = constraints.get("rank");
            String inputHeightValue = constraints.get("input_height");
            String inputWidthValue = constraints.get("input_width");
            String outputHeightValue = constraints.get("output_height");
            String outputWidthValue = constraints.get("output_width");
            if (dtype == null || rankValue == null || inputHeightValue == null
                    || inputWidthValue == null || outputHeightValue == null || outputWidthValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            Integer rank = parseInteger(rankValue);
            Integer inputHeight = parseInteger(inputHeightValue);
            Integer inputWidth = parseInteger(inputWidthValue);
            Integer outputHeight = parseInteger(outputHeightValue);
            Integer outputWidth = parseInteger(outputWidthValue);
            if (rank == null || inputHeight == null || inputWidth == null
                    || outputHeight == null || outputWidth == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (rank != 4 || inputHeight <= 0 || inputWidth <= 0
                    || outputHeight <= 0 || outputWidth <= 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * Cat metadata describes every input: comma-separated dtypes and
         * semicolon-separated shapes whose dimensions are comma-delimited.
         * Example: input_dtypes=F32,F32; input_shapes=2,3;2,4.
         */
        private OperationSupportDescriptor catSupport(
                String operationId, Map<String, String> constraints) {
            String dtypesValue = constraints.get("input_dtypes");
            String shapesValue = constraints.get("input_shapes");
            String axisValue = constraints.get("axis");
            if (dtypesValue == null || shapesValue == null || axisValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            String[] dtypes = dtypesValue.split(",", -1);
            String[] shapes = shapesValue.split(";", -1);
            Integer axis = parseInteger(axisValue);
            if (axis == null || dtypes.length == 0 || dtypes.length != shapes.length) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            int[][] dimensions = new int[shapes.length][];
            for (int i = 0; i < shapes.length; i++) {
                if (!isF32(dtypes[i])) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                String[] values = shapes[i].split(",", -1);
                dimensions[i] = new int[values.length];
                for (int d = 0; d < values.length; d++) {
                    Integer size = parseInteger(values[d]);
                    if (size == null) return new OperationSupportDescriptor(
                            operationId, OperationSupport.UNKNOWN, null);
                    if (size < 0) return new OperationSupportDescriptor(
                            operationId, OperationSupport.UNSUPPORTED, null);
                    dimensions[i][d] = size;
                }
            }
            int rank = dimensions[0].length;
            if (rank == 0 || axis < 0 || axis >= rank) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            for (int i = 1; i < dimensions.length; i++) {
                if (dimensions[i].length != rank) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                for (int d = 0; d < rank; d++) {
                    if (d != axis && dimensions[i][d] != dimensions[0][d]) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                }
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * CpuBackend GQA fallback requires rank-4 F32 Q/K/V, with query heads
         * divisible by key/value heads. Optional additive masks are F32 rank-2;
         * exact batch/sequence/head dimensions are checked by CpuBackend.
         */
        private OperationSupportDescriptor gqaSupport(
                String operationId, Map<String, String> constraints) {
            String queryDtype = constraints.get("dtype");
            String keyDtype = constraints.get("key_dtype");
            String valueDtype = constraints.get("value_dtype");
            String queryRankValue = constraints.get("rank");
            String keyRankValue = constraints.get("key_rank");
            String valueRankValue = constraints.get("value_rank");
            String queryHeadsValue = constraints.get("query_heads");
            String keyValueHeadsValue = constraints.get("key_value_heads");
            String causalValue = constraints.get("causal");
            String maskDtype = constraints.get("mask_dtype");
            String maskRankValue = constraints.get("mask_rank");

            if (queryDtype == null || keyDtype == null || valueDtype == null
                    || queryRankValue == null || keyRankValue == null || valueRankValue == null
                    || queryHeadsValue == null || keyValueHeadsValue == null || causalValue == null
                    || ((maskDtype == null) != (maskRankValue == null))) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(queryDtype) || !isF32(keyDtype) || !isF32(valueDtype)
                    || (maskDtype != null && !isF32(maskDtype))) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }

            Integer queryRank = parseInteger(queryRankValue);
            Integer keyRank = parseInteger(keyRankValue);
            Integer valueRank = parseInteger(valueRankValue);
            Integer queryHeads = parseInteger(queryHeadsValue);
            Integer keyValueHeads = parseInteger(keyValueHeadsValue);
            if (queryRank == null || keyRank == null || valueRank == null
                    || queryHeads == null || keyValueHeads == null
                    || !("true".equalsIgnoreCase(causalValue) || "false".equalsIgnoreCase(causalValue))) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (queryRank != 4 || keyRank != 4 || valueRank != 4
                    || queryHeads <= 0 || keyValueHeads <= 0
                    || queryHeads % keyValueHeads != 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            if (maskDtype != null) {
                Integer maskRank = parseInteger(maskRankValue);
                if (maskRank == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (maskRank != 2) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * The HAT backend exposes CpuBackend's simple RoPE overload:
         * [batch, heads, sequence, headDim], F32, even headDim, non-negative
         * position offset, and finite theta > 1. This does not advertise the
         * separate CPU overload that accepts explicit position IDs.
         */
        private OperationSupportDescriptor ropeSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            String rankValue = constraints.get("rank");
            String headDimValue = constraints.get("head_dim");
            String batchValue = constraints.get("batch");
            String headsValue = constraints.get("heads");
            String sequenceValue = constraints.get("sequence");
            String positionValue = constraints.get("position");
            String thetaValue = constraints.get("theta");
            if (dtype == null || rankValue == null || headDimValue == null
                    || batchValue == null || headsValue == null || sequenceValue == null
                    || positionValue == null || thetaValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            Integer rank = parseInteger(rankValue);
            Integer headDim = parseInteger(headDimValue);
            Integer batch = parseInteger(batchValue);
            Integer heads = parseInteger(headsValue);
            Integer sequence = parseInteger(sequenceValue);
            Integer position = parseInteger(positionValue);
            if (rank == null || headDim == null || batch == null || heads == null
                    || sequence == null || position == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (rank != 4 || headDim <= 0 || (headDim & 1) != 0
                    || batch <= 0 || heads <= 0 || sequence <= 0 || position < 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            float theta;
            try {
                theta = Float.parseFloat(thetaValue);
            } catch (NumberFormatException invalidTheta) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!Float.isFinite(theta) || theta <= 1.0f) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * CpuBackend.embedding reads F32 embedding rows and accepts integral
         * I32 or I8 token indices. The input rank is unrestricted; the output
         * appends the embedding dimension to the input shape.
         */
        private OperationSupportDescriptor embeddingSupport(
                String operationId, Map<String, String> constraints) {
            String inputDtype = constraints.get("dtype");
            String weightDtype = constraints.get("weight_dtype");
            String weightRankValue = constraints.get("weight_rank");
            if (inputDtype == null || weightDtype == null || weightRankValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!"I32".equalsIgnoreCase(inputDtype) && !"I8".equalsIgnoreCase(inputDtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            if (!isF32(weightDtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            try {
                if (Integer.parseInt(weightRankValue) != 2) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
            } catch (NumberFormatException invalidRank) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            return new OperationSupportDescriptor(
                    operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private OperationSupportDescriptor sliceSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            String rankValue = constraints.get("rank");
            if (dtype == null || rankValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            Integer rank = parseInteger(rankValue);
            if (rank == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            if (rank <= 0) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            Integer offsetsRank = parseInteger(constraints.get("offsets_rank"));
            Integer sizesRank = parseInteger(constraints.get("sizes_rank"));
            if (offsetsRank == null || sizesRank == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!offsetsRank.equals(rank) || !sizesRank.equals(rank)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private OperationSupportDescriptor transposeSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            String rankValue = constraints.get("rank");
            if (dtype == null || rankValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            Integer rank = parseInteger(rankValue);
            if (rank == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            if (rank <= 0) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            String dim0Value = constraints.get("dim0");
            String dim1Value = constraints.get("dim1");
            if (dim0Value == null && dim1Value == null) {
                return new OperationSupportDescriptor(operationId,
                        rank >= 2 ? OperationSupport.DELEGATED : OperationSupport.UNSUPPORTED,
                        rank >= 2 ? CPU_FALLBACK_BACKEND_ID : null);
            }
            if (dim0Value == null || dim1Value == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            Integer dim0 = parseInteger(dim0Value), dim1 = parseInteger(dim1Value);
            if (dim0 == null || dim1 == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (dim0 < 0 || dim1 < 0 || dim0 >= rank || dim1 >= rank) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private OperationSupportDescriptor splitSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            String rankValue = constraints.get("rank");
            if (dtype == null || rankValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            Integer rank = parseInteger(rankValue);
            Integer axis = parseInteger(constraints.get("axis"));
            Integer parts = parseInteger(constraints.get("parts"));
            Integer axisSize = parseInteger(constraints.get("axis_size"));
            if (rank == null || axis == null || parts == null || axisSize == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (rank <= 0 || axis < 0 || axis >= rank || parts <= 0
                    || axisSize <= 0 || parts > axisSize || axisSize % parts != 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private Integer parseInteger(String value) {
            if (value == null) return null;
            try {
                return Integer.valueOf(value);
            } catch (NumberFormatException invalid) {
                return null;
            }
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

        private OperationSupportDescriptor convolutionSupport(String operationId, Map<String, String> constraints) {
            OperationSupportDescriptor input = floatInputSupport(operationId, constraints, 4);
            if (input.support() != OperationSupport.DELEGATED) return input;
            String weightDtype = constraints.get("weight_dtype");
            if (weightDtype == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            if (!isF32(weightDtype)) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);

            OperationSupport bias = optionalNormalizationTensorSupport(constraints, "bias", false);
            if (bias != OperationSupport.DELEGATED) return new OperationSupportDescriptor(operationId, bias, null);
            for (String key : new String[]{"stride", "dilation"}) {
                OperationSupport support = positiveIntegerConstraint(constraints, key);
                if (support != OperationSupport.DELEGATED) return new OperationSupportDescriptor(operationId, support, null);
            }
            OperationSupport padding = nonNegativeIntegerConstraint(constraints, "padding");
            if (padding != OperationSupport.DELEGATED) return new OperationSupportDescriptor(operationId, padding, null);

            Long groups = parseLong(constraints.get("groups"));
            Long inputChannels = parseLong(constraints.get("input_channels"));
            Long outputChannels = parseLong(constraints.get("output_channels"));
            Long weightInputChannels = parseLong(constraints.get("weight_input_channels"));
            if (groups == null || inputChannels == null || outputChannels == null || weightInputChannels == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (groups <= 0 || inputChannels <= 0 || outputChannels <= 0 || weightInputChannels <= 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            if (inputChannels % groups != 0 || outputChannels % groups != 0
                    || weightInputChannels != inputChannels / groups) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private OperationSupportDescriptor poolingSupport(String operationId, Map<String, String> constraints,
                String firstKey, String secondKey, String nonNegativeKey) {
            OperationSupportDescriptor input = floatInputSupport(operationId, constraints, 4);
            if (input.support() != OperationSupport.DELEGATED) return input;

            OperationSupport first = positiveIntegerConstraint(constraints, firstKey);
            if (first != OperationSupport.DELEGATED) return new OperationSupportDescriptor(operationId, first, null);
            OperationSupport second = positiveIntegerConstraint(constraints, secondKey);
            if (second != OperationSupport.DELEGATED) return new OperationSupportDescriptor(operationId, second, null);
            if (nonNegativeKey != null) {
                OperationSupport padding = nonNegativeIntegerConstraint(constraints, nonNegativeKey);
                if (padding != OperationSupport.DELEGATED) return new OperationSupportDescriptor(operationId, padding, null);
            }

            Long inputHeight = parseLong(constraints.get("input_height"));
            Long inputWidth = parseLong(constraints.get("input_width"));
            if (inputHeight == null || inputWidth == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (inputHeight <= 0 || inputWidth <= 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }

            if ("tensor.max_pool2d".equals(operationId)) {
                long kernel = Long.parseLong(constraints.get(firstKey));
                long stride = Long.parseLong(constraints.get(secondKey));
                long padding = Long.parseLong(constraints.get(nonNegativeKey));
                try {
                    long paddedHeight = Math.addExact(inputHeight, Math.multiplyExact(2L, padding));
                    long paddedWidth = Math.addExact(inputWidth, Math.multiplyExact(2L, padding));
                    long outputHeight = (paddedHeight - kernel) / stride + 1;
                    long outputWidth = (paddedWidth - kernel) / stride + 1;
                    if (outputHeight <= 0 || outputWidth <= 0) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                } catch (ArithmeticException overflow) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
            }

            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private Long parseLong(String value) {
            if (value == null) return null;
            try {
                return Long.valueOf(value);
            } catch (NumberFormatException invalid) {
                return null;
            }
        }

        private OperationSupportDescriptor floatInputSupport(String operationId, Map<String, String> constraints, int requiredRank) {
            String dtype = constraints.get("dtype"), rank = constraints.get("rank");
            if (dtype == null || rank == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            if (!isF32(dtype)) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            try {
                if (Integer.parseInt(rank) != requiredRank) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            } catch (NumberFormatException invalidRank) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        private OperationSupport positiveIntegerConstraint(Map<String, String> constraints, String key) {
            String value = constraints.get(key);
            if (value == null) return OperationSupport.UNKNOWN;
            try { return Integer.parseInt(value) > 0 ? OperationSupport.DELEGATED : OperationSupport.UNSUPPORTED; }
            catch (NumberFormatException invalid) { return OperationSupport.UNKNOWN; }
        }

        private OperationSupport nonNegativeIntegerConstraint(Map<String, String> constraints, String key) {
            String value = constraints.get(key);
            if (value == null) return OperationSupport.UNKNOWN;
            try { return Integer.parseInt(value) >= 0 ? OperationSupport.DELEGATED : OperationSupport.UNSUPPORTED; }
            catch (NumberFormatException invalid) { return OperationSupport.UNKNOWN; }
        }

        /**
         * CpuBackend softmax uses a non-negative axis directly; it does not
         * normalize negative dimensions. The no-axis softmax overload selects
         * the final dimension, while log_softmax requires an explicit axis.
         */
        private OperationSupportDescriptor softmaxSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            String rankValue = constraints.get("rank");
            if (dtype == null || rankValue == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            int rank;
            try {
                rank = Integer.parseInt(rankValue);
            } catch (NumberFormatException invalidRank) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (rank <= 0) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }

            String dimension = constraints.get("dim");
            if (dimension == null) {
                if ("tensor.log_softmax".equals(operationId)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
            }
            try {
                int axis = Integer.parseInt(dimension);
                if (axis < 0 || axis >= rank) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
            } catch (NumberFormatException invalidDimension) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * Models global and axis reductions exposed by CpuBackend. The CPU
         * backend supports axis sum/mean (including negative axes), but max
         * currently has only a global-reduction overload.
         */
        private OperationSupportDescriptor reductionSupport(
                String operationId, Map<String, String> constraints) {
            String dtype = constraints.get("dtype");
            if (dtype == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!isF32(dtype)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }

            String rankValue = constraints.get("rank");
            Integer rank = null;
            if (rankValue != null) {
                try {
                    rank = Integer.valueOf(rankValue);
                    if (rank < 0) throw new NumberFormatException("negative rank");
                } catch (NumberFormatException invalidRank) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
            }

            String dimension = constraints.get("dim");
            if (dimension != null) {
                if ("tensor.max".equals(operationId)) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                if (rank == null) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                if (rank == 0) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                }
                try {
                    int axis = Integer.parseInt(dimension);
                    if (axis < 0) axis += rank;
                    if (axis < 0 || axis >= rank) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                } catch (NumberFormatException invalidDimension) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
                String dimensionSize = constraints.get("dim_size");
                if (dimensionSize != null) {
                    try {
                        long size = Long.parseLong(dimensionSize);
                        if (size < 0) throw new NumberFormatException("negative dimension size");
                        if (size == 0 && "tensor.mean".equals(operationId)) {
                            return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                        }
                    } catch (NumberFormatException invalidDimensionSize) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                    }
                }
            }

            String keepDim = constraints.get("keep_dim");
            if (keepDim != null && !"true".equalsIgnoreCase(keepDim)
                    && !"false".equalsIgnoreCase(keepDim)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }

            String numelValue = constraints.get("numel");
            if (numelValue != null) {
                try {
                    long numel = Long.parseLong(numelValue);
                    if (numel < 0) throw new NumberFormatException("negative numel");
                    if (numel == 0 && dimension == null
                            && ("tensor.mean".equals(operationId) || "tensor.max".equals(operationId))) {
                        return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
                    }
                } catch (NumberFormatException invalidNumel) {
                    return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
                }
            }
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
        }

        /**
         * Inference/zero-probability dropout is an identity operation. The
         * training kernel reads and writes F32 values and requires 0 <= p < 1.
         */
        private OperationSupportDescriptor dropoutSupport(
                String operationId, Map<String, String> constraints) {
            String training = constraints.get("training");
            String probability = constraints.get("p");
            if (training == null || probability == null) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!"true".equalsIgnoreCase(training) && !"false".equalsIgnoreCase(training)) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            float p;
            try {
                p = Float.parseFloat(probability);
            } catch (NumberFormatException invalidProbability) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            }
            if (!Float.isFinite(p) || p < 0f || p > 1f) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            if (!Boolean.parseBoolean(training) || p == 0f) {
                // Identity dropout is handled by the selected backend without
                // executing a CPU kernel or changing tensor ownership.
                return new OperationSupportDescriptor(operationId, OperationSupport.NATIVE, null);
            }
            if (p == 1f) {
                return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            }
            String dtype = constraints.get("dtype");
            if (dtype == null) return new OperationSupportDescriptor(operationId, OperationSupport.UNKNOWN, null);
            if (!isF32(dtype)) return new OperationSupportDescriptor(operationId, OperationSupport.UNSUPPORTED, null);
            return new OperationSupportDescriptor(operationId, OperationSupport.DELEGATED, CPU_FALLBACK_BACKEND_ID);
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
