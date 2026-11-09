package tech.kayys.aljabr.backend.metal;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.metal.binding.MetalBinding;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.memory.TensorAllocator;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntSupplier;
import java.util.function.Supplier;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Metal hardware-accelerated computation backend.
 */
public class MetalComputeBackend implements ComputeBackend {

    /**
     * Re-wraps a short-lived CPU fallback result with this backend identity.
     *
     * <p>The returned tensor adopts the result's existing storage reference;
     * no additional retain is performed because the fallback result is an
     * internal intermediate and must not be released separately.</p>
     */
    private Tensor wrap(Tensor cpuRes) {
        Objects.requireNonNull(cpuRes, "CPU fallback result must not be null");
        try {
            TensorStorage storage = Objects.requireNonNull(
                    cpuRes.storage(), "CPU fallback tensor storage must not be null");
            return new DefaultTensor(cpuRes.shape(), cpuRes.dtype(), cpuRes.device(), storage, this);
        } catch (UnsupportedOperationException storageNotExposed) {
            // Preserve support for Tensor implementations that do not expose
            // backing storage through the optional Tensor.storage() capability.
            return cpuRes;
        }
    }

    private java.util.List<Tensor> wrapList(java.util.List<Tensor> list) {
        java.util.List<Tensor> res = new java.util.ArrayList<>(list.size());
        for (Tensor t : list) {
            res.add(wrap(t));
        }
        return res;
    }

    private static final Logger LOG = Logger.getLogger(MetalComputeBackend.class);
    private static final String FORCE_CPU_PROPERTY = "aljabr.kernel.force.cpu";

    private final MetalBinding metalBinding;
    private final CpuBackend cpuFallback;
    private final boolean isNative;
    private final boolean forceCpu;
    private final TensorAllocator hostStagingAllocator;
    private final MetalFallbackPolicy fallbackPolicy;
    private final MetalExecutionObserver executionObserver;

    public MetalComputeBackend() {
        this(null, MetalRuntimePolicy.ALLOW_CPU_FALLBACK, MetalFallbackPolicy.ALLOW_CPU_FALLBACK, MetalExecutionObserver.NO_OP);
    }

    /**
     * Creates a backend with an explicit allocator for host-accessible output staging.
     */
    public MetalComputeBackend(TensorAllocator hostStagingAllocator) {
        this(hostStagingAllocator, MetalRuntimePolicy.ALLOW_CPU_FALLBACK, MetalFallbackPolicy.ALLOW_CPU_FALLBACK, MetalExecutionObserver.NO_OP);
    }

    /**
     * REQUIRE_NATIVE rejects forced-CPU mode, a missing native library, or an
     * unavailable Metal device. It guarantees runtime initialization only;
     * individual operations may still use CPU kernels when no native kernel exists.
     */
    public MetalComputeBackend(TensorAllocator hostStagingAllocator, MetalRuntimePolicy runtimePolicy) {
        this(hostStagingAllocator, runtimePolicy, MetalFallbackPolicy.ALLOW_CPU_FALLBACK, MetalExecutionObserver.NO_OP);
    }

    /**
     * Creates a backend with independent runtime-initialization and per-operation fallback policies.
     * FAIL_IF_UNSUPPORTED rejects every operation that would otherwise execute through the CPU backend.
     */
    public MetalComputeBackend(
            TensorAllocator hostStagingAllocator,
            MetalRuntimePolicy runtimePolicy,
            MetalFallbackPolicy fallbackPolicy) {
        this(hostStagingAllocator, runtimePolicy, fallbackPolicy, MetalExecutionObserver.NO_OP);
    }

    /**
     * Creates a backend with opt-in execution reporting. Observer callbacks are isolated
     * from computation: observer failures are logged and never change operation results.
     */
    public MetalComputeBackend(
            TensorAllocator hostStagingAllocator,
            MetalRuntimePolicy runtimePolicy,
            MetalFallbackPolicy fallbackPolicy,
            MetalExecutionObserver executionObserver) {
        this.hostStagingAllocator = hostStagingAllocator != null ? hostStagingAllocator : CpuBuffer::new;
        this.fallbackPolicy = Objects.requireNonNull(fallbackPolicy, "fallbackPolicy");
        this.executionObserver = Objects.requireNonNull(executionObserver, "executionObserver");
        Objects.requireNonNull(runtimePolicy, "runtimePolicy");
        this.forceCpu = Boolean.parseBoolean(System.getProperty(FORCE_CPU_PROPERTY, "false"));

        if (forceCpu) {
            if (runtimePolicy == MetalRuntimePolicy.REQUIRE_NATIVE) {
                throw new IllegalStateException("Native Metal runtime is required, but CPU mode was forced");
            }
            LOG.info("MetalComputeBackend forced into CPU mode by system properties.");
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.isNative = false;
            this.cpuFallback = new CpuBackend();
            return;
        }

        boolean loaded = MetalBinding.initialize();
        if (!loaded && runtimePolicy == MetalRuntimePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native Metal runtime is required, but the Metal library could not be loaded");
        }
        if (!loaded) {
            LOG.warn("Failed to initialize MetalBinding. MetalComputeBackend will operate in CPU fallback mode.");
            MetalBinding.initializeFallback();
        }

        this.metalBinding = MetalBinding.getInstance();
        this.metalBinding.init();
        this.isNative = metalBinding.isRuntimeActive();
        if (!isNative && runtimePolicy == MetalRuntimePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native Metal runtime is required, but no active Metal device was initialized");
        }
        this.cpuFallback = new CpuBackend();
        LOG.infof("Initialized MetalComputeBackend [Device: %s, Unified Memory: %s]",
                metalBinding.deviceName(), metalBinding.isUnifiedMemory());
    }

    /**
     * Exposes storage to the current native Metal FFM bridge.
     *
     * <p>The current bridge accepts host MemorySegments. Opaque device-local storage must not be
     * passed through this boundary until Metal-native buffer binding and
     * transfer operations are implemented.</p>
     */
    static MemorySegment metalSegment(Tensor tensor) {
        try {
            return MetalTensorStorageAccess.segment(tensor);
        } catch (UnsupportedOperationException failure) {
            throw new UnsupportedOperationException(
                    "Metal FFM dispatch requires host-accessible storage; opaque device-local storage "
                            + "requires a Metal-native binding path",
                    failure);
        }
    }

    private long byteSize(DType dtype) {
        return switch (dtype) {
            case F32, I32 -> 4;
            case F16, BF16 -> 2;
            case I8, INT8, Q8_0 -> 1;
            case Q4_K, Q4_0 -> 0;
        };
    }

    /**
     * MetalBinding currently accepts host-addressable MemorySegments, so
     * outputs are host-accessible staging storage. This is deliberately not
     * represented as device-local Metal memory; a native device allocator
     * requires a separate TensorStorage implementation and binding contract.
     */
    private HostAccessibleStorage allocateHostStorage(long sizeBytes) {
        TensorStorage storage = Objects.requireNonNull(
                hostStagingAllocator.allocate(sizeBytes),
                "Metal host-staging allocator must not return null");
        if (!(storage instanceof HostAccessibleStorage hostStorage)) {
            storage.release();
            throw new IllegalArgumentException(
                    "Metal FFM output staging requires HostAccessibleStorage, got "
                            + storage.getClass().getName());
        }
        if (hostStorage.sizeBytes() < sizeBytes) {
            long available = hostStorage.sizeBytes();
            hostStorage.release();
            throw new IllegalStateException(
                    "Metal host-staging allocator returned " + available
                            + " bytes; " + sizeBytes + " required");
        }
        return hostStorage;
    }

    /**
     * Runs a native operation that writes into a newly allocated output buffer.
     * Ownership transfers to the result tensor only after a successful dispatch.
     * On native failure or an exception, the temporary output is released before
     * the CPU fallback is evaluated or the exception is propagated.
     */
    static Tensor dispatchOutput(
            HostAccessibleStorage output,
            IntSupplier dispatch,
            Supplier<Tensor> fallback,
            Function<HostAccessibleStorage, Tensor> resultFactory) {
        Objects.requireNonNull(output, "output must not be null");
        Objects.requireNonNull(dispatch, "dispatch must not be null");
        Objects.requireNonNull(fallback, "fallback must not be null");
        Objects.requireNonNull(resultFactory, "resultFactory must not be null");

        final int status;
        try {
            status = dispatch.getAsInt();
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }

        if (status != 0) {
            output.release();
            return fallback.get();
        }

        try {
            return resultFactory.apply(output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }

    /**
     * Executes a CPU implementation only when the configured fallback policy permits it.
     * This is the single gate for unsupported native operations and failed native dispatches.
     */
    private <T> T fallback(String operation, Supplier<T> cpuExecution) {
        if (fallbackPolicy == MetalFallbackPolicy.FAIL_IF_UNSUPPORTED) {
            reportExecution(new MetalExecutionEvent(operation, MetalExecutionEvent.ExecutionPath.REJECTED,
                    "Native Metal execution is unavailable for this operation or input"));
            throw new UnsupportedOperationException(
                    "Metal operation '" + operation
                            + "' cannot execute natively and CPU fallback is disabled");
        }
        reportExecution(new MetalExecutionEvent(operation, MetalExecutionEvent.ExecutionPath.CPU_FALLBACK,
                "No usable native Metal implementation for this operation or input"));
        LOG.debugf("Metal operation '%s' is executing through the CPU fallback", operation);
        return cpuExecution.get();
    }

    private void reportExecution(MetalExecutionEvent event) {
        try {
            executionObserver.onExecution(event);
        } catch (RuntimeException observerFailure) {
            LOG.warnf(observerFailure, "Metal execution observer failed for operation '%s'", event.operation());
        }
    }

    private Tensor dispatchOutputObserved(
            String operation,
            HostAccessibleStorage output,
            IntSupplier dispatch,
            Supplier<Tensor> fallback,
            Function<HostAccessibleStorage, Tensor> resultFactory) {
        AtomicBoolean nativeDispatchSucceeded = new AtomicBoolean();
        Tensor result = dispatchOutput(output, () -> {
            int status = dispatch.getAsInt();
            nativeDispatchSucceeded.set(status == 0);
            return status;
        }, fallback, resultFactory);
        if (nativeDispatchSucceeded.get()) {
            reportExecution(new MetalExecutionEvent(operation, MetalExecutionEvent.ExecutionPath.METAL_NATIVE,
                    "Native Metal dispatch completed successfully"));
        }
        return result;
    }

    private static HostAccessibleStorage hostStorage(Tensor tensor) {
        return MetalTensorStorageAccess.hostStorage(tensor);
    }

    private static MemorySegment segment(Tensor tensor) {
        return hostStorage(tensor).segment();
    }

    private static MemorySegment optionalSegment(Tensor tensor) {
        return tensor == null ? MemorySegment.NULL : segment(tensor);
    }

    @Override
    public DeviceType deviceType() {
        return isNative ? DeviceType.METAL : DeviceType.CPU;
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return fallback("add", () -> wrap(cpuFallback.add(a, b)));
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        return fallback("sub", () -> wrap(cpuFallback.sub(a, b)));
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        return fallback("mul", () -> wrap(cpuFallback.mul(a, scalar)));
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        return fallback("mul", () -> wrap(cpuFallback.mul(a, b)));
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return fallback("div", () -> wrap(cpuFallback.div(a, scalar)));
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        return fallback("div", () -> wrap(cpuFallback.div(a, b)));
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        return fallback("addScalar", () -> wrap(cpuFallback.addScalar(a, scalar)));
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (!isNative)
            return fallback("matmul", () -> wrap(cpuFallback.matmul(a, b)));


        if (b.dtype() == tech.kayys.aljabr.core.tensor.DType.Q4_K
                || b.dtype() == tech.kayys.aljabr.core.tensor.DType.Q8_0) {
            return matmulQuantized(a, b);
        }

        int M = (int) a.shape().dim(a.shape().rank() - 2);
        int K = (int) a.shape().dim(a.shape().rank() - 1);
        int N = (int) b.shape().dim(b.shape().rank() - 1);

        Shape shapeC = new Shape(M, N);
        long sizeBytes = shapeC.numel() * byteSize(a.dtype());

        HostAccessibleStorage bufferC = allocateHostStorage(sizeBytes);
        return dispatchOutputObserved("matmul",
                bufferC,
                () -> metalBinding.matmul(bufferC.segment(), segment(a), segment(b), M, K, N, 1.0f, 0.0f),
                () -> fallback("matmul", () -> wrap(cpuFallback.matmul(a, b))),
                output -> new DefaultTensor(shapeC, a.dtype(), a.device(), output, this));
    }

    private Tensor matmulQuantized(Tensor a, Tensor db) {
        int M = (int) a.shape().dim(a.shape().rank() - 2);
        int K = (int) a.shape().dim(a.shape().rank() - 1);
        int N = (int) db.shape().dim(0); // weight rows = output dim

        Shape shapeC = new Shape(M, N);
        HostAccessibleStorage bufferC = allocateHostStorage((long) M * N * 4);

        return dispatchOutputObserved("matmul",
                bufferC,
                () -> {
                    int status = 0;
                    if (db.dtype() == tech.kayys.aljabr.core.tensor.DType.Q4_K) {
                        for (int m = 0; m < M; m++) {
                            status = metalBinding.matvecTransposedRightQ4K(
                                    bufferC.segment().asSlice(m * N * 4L, N * 4L),
                                    segment(a).asSlice(m * K * 4L, K * 4L),
                                    segment(db), K, N);
                            if (status != 0) break;
                        }
                    } else if (db.dtype() == tech.kayys.aljabr.core.tensor.DType.Q8_0) {
                        for (int m = 0; m < M; m++) {
                            status = metalBinding.matvecTransposedRightQ8_0(
                                    bufferC.segment().asSlice(m * N * 4L, N * 4L),
                                    segment(a).asSlice(m * K * 4L, K * 4L),
                                    segment(db), K, N);
                            if (status != 0) break;
                        }
                    } else {
                        status = -1;
                    }
                    return status;
                },
                () -> fallback("matmul", () -> wrap(cpuFallback.matmul(a, db))),
                output -> new DefaultTensor(shapeC, tech.kayys.aljabr.core.tensor.DType.F32, a.device(), output, this));
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return fallback("reshape", () -> wrap(cpuFallback.reshape(a, newShape)));
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V, Tensor additiveMask, boolean causal) {
        if (additiveMask == null && !causal) {
            return attention(Q, K, V);
        }
        // Masked attention currently uses the CPU reference implementation.
        // The result is wrapped to preserve this backend's tensor ownership contract.
        return fallback("attention", () -> wrap(cpuFallback.attention(Q, K, V, additiveMask, causal)));
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        if (!isNative)
            return fallback("attention", () -> wrap(cpuFallback.attention(Q, K, V)));

        int B = (int) Q.shape().dim(0);
        int T = (int) Q.shape().dim(1);
        int H = (int) Q.shape().dim(2);
        int Hkv = (int) K.shape().dim(2);
        int D = (int) Q.shape().dim(3);
        int Skv = (int) K.shape().dim(1);

        Shape shapeOut = Q.shape();
        long sizeBytes = shapeOut.numel() * byteSize(Q.dtype());
        HostAccessibleStorage bufferOut = allocateHostStorage(sizeBytes);

        return dispatchOutputObserved("attention",
                bufferOut,
                () -> {
                    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
                        java.lang.foreign.MemorySegment contextLens =
                                arena.allocate(java.lang.foreign.ValueLayout.JAVA_INT, B);
                        for (int b = 0; b < B; b++) {
                            contextLens.setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, b, Skv);
                        }
                        java.lang.foreign.MemorySegment empty = java.lang.foreign.MemorySegment.NULL;
                        if (H == Hkv) {
                            return metalBinding.flashAttention(bufferOut.segment(), segment(Q), segment(K),
                                    segment(V), B, T, H, D);
                        }
                        return metalBinding.attentionGqa(bufferOut.segment(), segment(Q), segment(K),
                                segment(V), empty, contextLens, B, T, H, Hkv, D, 16, 1024,
                                (float) (1.0 / Math.sqrt(D)), 1, 0.0f);
                    }
                },
                () -> fallback("attention", () -> wrap(cpuFallback.attention(Q, K, V))),
                output -> new DefaultTensor(shapeOut, Q.dtype(), Q.device(), output, this));
    }

    @Override
    public Tensor softmax(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return fallback("softmax", () -> wrap(cpuFallback.softmax(a)));

        Shape shape = a.shape();
        int n = (int) shape.numel();
        HostAccessibleStorage bufferOut = allocateHostStorage(n * 4);

        return dispatchOutputObserved("softmax",
                bufferOut,
                () -> metalBinding.softmax(bufferOut.segment(), segment(a), n),
                () -> fallback("softmax", () -> wrap(cpuFallback.softmax(a))),
                output -> new DefaultTensor(shape, a.dtype(), a.device(), output, this));
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        return fallback("slice", () -> wrap(cpuFallback.slice(a, offsets, sizes)));
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        return fallback("split", () -> wrapList(cpuFallback.split(a, axis, parts)));
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        return fallback("pow", () -> wrap(cpuFallback.pow(a, exponent)));
    }

    @Override
    public Tensor mean(Tensor a) {
        return fallback("mean", () -> wrap(cpuFallback.mean(a)));
    }

    @Override
    public Tensor abs(Tensor a) {
        return fallback("abs", () -> wrap(cpuFallback.abs(a)));
    }

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        return fallback("crossEntropy", () -> wrap(cpuFallback.crossEntropy(pred, target)));
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        return fallback("binaryCrossEntropy", () -> wrap(cpuFallback.binaryCrossEntropy(pred, target)));
    }

    @Override
    public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
        return fallback("cast", () -> wrap(cpuFallback.cast(a, dtype)));
    }

    @Override
    public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
        if (device == DeviceType.METAL || device == DeviceType.CPU) {
            return a;
        }
        return fallback("to", () -> wrap(cpuFallback.to(a, device)));
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        Shape shape = a.shape();
        long sizeBytes = shape.numel() * byteSize(a.dtype());
        HostAccessibleStorage buffer = allocateHostStorage(sizeBytes);
        return new DefaultTensor(shape, a.dtype(), a.device(), buffer, this);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        return fallback("sqrt", () -> wrap(cpuFallback.sqrt(a)));
    }

    @Override
    public Tensor relu(Tensor a) {
        return fallback("relu", () -> wrap(cpuFallback.relu(a)));
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        return fallback("sigmoid", () -> wrap(cpuFallback.sigmoid(a)));
    }

    @Override
    public Tensor tanh(Tensor a) {
        return fallback("tanh", () -> wrap(cpuFallback.tanh(a)));
    }

    @Override
    public Tensor log(Tensor a) {
        return fallback("log", () -> wrap(cpuFallback.log(a)));
    }

    @Override
    public Tensor exp(Tensor a) {
        return fallback("exp", () -> wrap(cpuFallback.exp(a)));
    }

    @Override
    public Tensor silu(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return fallback("silu", () -> wrap(cpuFallback.silu(a)));

        Shape shape = a.shape();
        int n = (int) shape.numel();
        HostAccessibleStorage bufferOut = allocateHostStorage(n * 4);

        return dispatchOutputObserved("silu",
                bufferOut,
                () -> metalBinding.silu(bufferOut.segment(), segment(a), n),
                () -> fallback("silu", () -> wrap(cpuFallback.silu(a))),
                output -> new DefaultTensor(shape, a.dtype(), a.device(), output, this));
    }

    @Override
    public Tensor flatten(Tensor a) {
        return fallback("flatten", () -> wrap(cpuFallback.flatten(a)));
    }

    @Override
    public Tensor unsqueeze(Tensor a, int dim) {
        return fallback("unsqueeze", () -> wrap(cpuFallback.unsqueeze(a, dim)));
    }

    @Override
    public Tensor squeeze(Tensor a) {
        return fallback("squeeze", () -> wrap(cpuFallback.squeeze(a)));
    }

    @Override
    public Tensor transpose(Tensor a) {
        return fallback("transpose", () -> wrap(cpuFallback.transpose(a)));
    }

    @Override
    public Tensor transpose(Tensor a, int d0, int d1) {
        return fallback("transpose", () -> wrap(cpuFallback.transpose(a, d0, d1)));
    }

    @Override
    public Tensor gelu(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return fallback("gelu", () -> wrap(cpuFallback.gelu(a)));

        Shape shape = a.shape();
        int n = (int) shape.numel();
        HostAccessibleStorage bufferOut = allocateHostStorage(n * 4);

        return dispatchOutputObserved("gelu",
                bufferOut,
                () -> metalBinding.gelu(bufferOut.segment(), segment(a), n),
                () -> fallback("gelu", () -> wrap(cpuFallback.gelu(a))),
                output -> new DefaultTensor(shape, a.dtype(), a.device(), output, this));
    }

    @Override
    public Tensor softmax(Tensor a, int dim) {
        if (!isNative || a.dtype() != DType.F32)
            return fallback("softmax", () -> wrap(cpuFallback.softmax(a, dim)));

        if (dim == a.shape().rank() - 1) {
            int rows = 1;
            for (int i = 0; i < dim; i++) {
                rows *= a.shape().dim(i);
            }
            int cols = (int) a.shape().dim(dim);
            final int rowCount = rows;

            HostAccessibleStorage bufferOut = allocateHostStorage(rowCount * cols * 4L);
            return dispatchOutputObserved("softmax",
                    bufferOut,
                    () -> metalBinding.softmaxRows(bufferOut.segment(), segment(a), rowCount, cols),
                    () -> fallback("softmax", () -> wrap(cpuFallback.softmax(a, dim))),
                    output -> new DefaultTensor(a.shape(), a.dtype(), a.device(), output, this));
        }
        return fallback("softmax", () -> wrap(cpuFallback.softmax(a, dim)));
    }

    @Override
    public Tensor logSoftmax(Tensor a, int dim) {
        return fallback("logSoftmax", () -> wrap(cpuFallback.logSoftmax(a, dim)));
    }

    @Override
    public Tensor mean(Tensor a, int dim, boolean keepDim) {
        return fallback("mean", () -> wrap(cpuFallback.mean(a, dim, keepDim)));
    }

    @Override
    public Tensor sum(Tensor a) {
        return fallback("sum", () -> wrap(cpuFallback.sum(a)));
    }

    @Override
    public Tensor sum(Tensor a, int dim, boolean keepDim) {
        return fallback("sum", () -> wrap(cpuFallback.sum(a, dim, keepDim)));
    }

    @Override
    public Tensor max(Tensor a) {
        return fallback("max", () -> wrap(cpuFallback.max(a)));
    }

    @Override
    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        if (!isNative || input.dtype() != DType.F32)
            return fallback("layerNorm", () -> wrap(cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps)));

        Shape shape = input.shape();
        long sizeBytes = shape.numel() * byteSize(input.dtype());
        HostAccessibleStorage bufferOut = allocateHostStorage(sizeBytes);

        int n = 1;
        for (long s : normalizedShape) {
            n *= s;
        }
        int rows = (int) (shape.numel() / n);
        final int normalizedWidth = n;
        final int rowCount = rows;

        return dispatchOutputObserved("layerNorm",
                bufferOut,
                () -> rowCount == 1
                        ? metalBinding.layerNorm(bufferOut.segment(), segment(input),
                                optionalSegment(weight), optionalSegment(bias), normalizedWidth, eps)
                        : metalBinding.layerNormRows(bufferOut.segment(), segment(input),
                                optionalSegment(weight), optionalSegment(bias), rowCount, normalizedWidth, eps),
                () -> fallback("layerNorm", () -> wrap(cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps))),
                output -> new DefaultTensor(shape, input.dtype(), input.device(), output, this));
    }

    @Override
    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        if (!isNative || input.dtype() != DType.F32)
            return fallback("rmsNorm", () -> wrap(cpuFallback.rmsNorm(input, weight, eps)));

        Shape shape = input.shape();
        int n = (int) shape.dim(shape.rank() - 1);
        int rows = 1;
        for (int i = 0; i < shape.rank() - 1; i++) {
            rows *= shape.dim(i);
        }
        final int normalizedWidth = n;
        final int rowCount = rows;
        HostAccessibleStorage bufferOut = allocateHostStorage((long) rowCount * normalizedWidth * 4);
        return dispatchOutputObserved("rmsNorm",
                bufferOut,
                () -> rowCount == 1
                        ? metalBinding.rmsNorm(bufferOut.segment(), segment(input), segment(weight), normalizedWidth, eps, false)
                        : metalBinding.rmsNormRows(bufferOut.segment(), segment(input),
                                segment(weight), rowCount, normalizedWidth, eps, false),
                () -> fallback("rmsNorm", () -> wrap(cpuFallback.rmsNorm(input, weight, eps))),
                output -> new DefaultTensor(shape, input.dtype(), input.device(), output, this));
    }

    @Override
    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        return fallback("batchNorm", () -> wrap(cpuFallback.batchNorm(input, weight, bias, runningMean, runningVar, training, momentum, eps)));
    }

    @Override
    public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return fallback("conv2d", () -> wrap(cpuFallback.conv2d(input, weight, bias, stride, padding, dilation, groups)));
    }

    @Override
    public Tensor upsampleNearest2d(Tensor input, int outputH, int outputW) {
        return fallback("upsampleNearest2d", () -> wrap(cpuFallback.upsampleNearest2d(input, outputH, outputW)));
    }

    @Override
    public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        return fallback("maxPool2d", () -> wrap(cpuFallback.maxPool2d(input, kernelSize, stride, padding)));
    }

    @Override
    public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        return fallback("adaptiveAvgPool2d", () -> wrap(cpuFallback.adaptiveAvgPool2d(input, outputH, outputW)));
    }

    @Override
    public Tensor dropout(Tensor input, float p, boolean training) {
        return fallback("dropout", () -> wrap(cpuFallback.dropout(input, p, training)));
    }

    @Override
    public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
        return fallback("embedding", () -> wrap(cpuFallback.embedding(weight, input, paddingIdx)));
    }

    @Override
    public Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        if (!isNative || input.dtype() != DType.F32) {
            return fallback("applyRoPE", () -> wrap(cpuFallback.applyRoPE(input, posOffset, freqBase, isNeox)));
        }

        int n = Math.toIntExact(input.numel());
        int headDim = Math.toIntExact(input.shape().dim(input.shape().rank() - 1)); // Last dimension is headDim

        HostAccessibleStorage bufferOut = allocateHostStorage((long) n * 4);
        return dispatchOutputObserved("applyRoPE",
                bufferOut,
                () -> metalBinding.rope(bufferOut.segment(), segment(input), n, headDim, posOffset, freqBase,
                        isNeox),
                () -> fallback("applyRoPE", () -> wrap(cpuFallback.applyRoPE(input, posOffset, freqBase, isNeox))),
                output -> new DefaultTensor(input.shape(), DType.F32, DeviceType.METAL, output, this));
    }


    @Override
    public Tensor cat(List<Tensor> tensors, int axis) {
        return fallback("cat", () -> wrap(cpuFallback.cat(tensors, axis)));
    }

    @Override
    public long numel(Tensor a) {
        return cpuFallback.numel(a);
    }
}
