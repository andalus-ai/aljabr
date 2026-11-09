package tech.kayys.aljabr.backend.metal;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.metal.binding.MetalBinding;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntSupplier;
import java.util.function.Supplier;

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

    public MetalComputeBackend() {
        this.forceCpu = Boolean.parseBoolean(System.getProperty(FORCE_CPU_PROPERTY, "false"));

        if (forceCpu) {
            LOG.info("MetalComputeBackend forced into CPU mode by system properties.");
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.isNative = false;
            this.cpuFallback = new CpuBackend();
            return;
        }

        boolean loaded = MetalBinding.initialize();
        if (!loaded) {
            LOG.warn("Failed to initialize MetalBinding. MetalComputeBackend will operate in CPU fallback mode.");
            MetalBinding.initializeFallback();
        }

        this.metalBinding = MetalBinding.getInstance();
        this.metalBinding.init();
        this.isNative = metalBinding.isRuntimeActive();
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
        return new CpuBuffer(sizeBytes);
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
        return wrap(cpuFallback.add(a, b));
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        return wrap(cpuFallback.sub(a, b));
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        return wrap(cpuFallback.mul(a, scalar));
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        return wrap(cpuFallback.mul(a, b));
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return wrap(cpuFallback.div(a, scalar));
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        return wrap(cpuFallback.div(a, b));
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        return wrap(cpuFallback.addScalar(a, scalar));
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (!isNative)
            return wrap(cpuFallback.matmul(a, b));


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
        return dispatchOutput(
                bufferC,
                () -> metalBinding.matmul(bufferC.segment(), segment(a), segment(b), M, K, N, 1.0f, 0.0f),
                () -> {
                    System.err.println("Metal matmul falling back to CPU for a=" + a.dtype()
                            + " (shape=" + a.shape() + "), b=" + b.dtype() + " (shape=" + b.shape()
                            + "), M=" + M + ", K=" + K + ", N=" + N);
                    return wrap(cpuFallback.matmul(a, b));
                },
                output -> new DefaultTensor(shapeC, a.dtype(), a.device(), output, this));
    }

    private Tensor matmulQuantized(Tensor a, Tensor db) {
        int M = (int) a.shape().dim(a.shape().rank() - 2);
        int K = (int) a.shape().dim(a.shape().rank() - 1);
        int N = (int) db.shape().dim(0); // weight rows = output dim

        Shape shapeC = new Shape(M, N);
        HostAccessibleStorage bufferC = allocateHostStorage((long) M * N * 4);

        return dispatchOutput(
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
                () -> {
                    System.err.println("Metal matmul falling back to CPU for a=" + a.dtype()
                            + " (shape=" + a.shape() + "), b=" + db.dtype() + " (shape=" + db.shape()
                            + "), M=" + M + ", K=" + K + ", N=" + N);
                    return wrap(cpuFallback.matmul(a, db));
                },
                output -> new DefaultTensor(shapeC, tech.kayys.aljabr.core.tensor.DType.F32, a.device(), output, this));
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return wrap(cpuFallback.reshape(a, newShape));
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V, Tensor additiveMask, boolean causal) {
        if (additiveMask == null && !causal) {
            return attention(Q, K, V);
        }
        // Masked attention currently uses the CPU reference implementation.
        // The result is wrapped to preserve this backend's tensor ownership contract.
        return wrap(cpuFallback.attention(Q, K, V, additiveMask, causal));
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        if (!isNative)
            return wrap(cpuFallback.attention(Q, K, V));

        int B = (int) Q.shape().dim(0);
        int T = (int) Q.shape().dim(1);
        int H = (int) Q.shape().dim(2);
        int Hkv = (int) K.shape().dim(2);
        int D = (int) Q.shape().dim(3);
        int Skv = (int) K.shape().dim(1);

        Shape shapeOut = Q.shape();
        long sizeBytes = shapeOut.numel() * byteSize(Q.dtype());
        HostAccessibleStorage bufferOut = allocateHostStorage(sizeBytes);

        return dispatchOutput(
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
                () -> wrap(cpuFallback.attention(Q, K, V)),
                output -> new DefaultTensor(shapeOut, Q.dtype(), Q.device(), output, this));
    }

    @Override
    public Tensor softmax(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return wrap(cpuFallback.softmax(a));

        Shape shape = a.shape();
        int n = (int) shape.numel();
        HostAccessibleStorage bufferOut = allocateHostStorage(n * 4);

        return dispatchOutput(
                bufferOut,
                () -> metalBinding.softmax(bufferOut.segment(), segment(a), n),
                () -> wrap(cpuFallback.softmax(a)),
                output -> new DefaultTensor(shape, a.dtype(), a.device(), output, this));
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        return wrap(cpuFallback.slice(a, offsets, sizes));
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        return wrapList(cpuFallback.split(a, axis, parts));
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        return wrap(cpuFallback.pow(a, exponent));
    }

    @Override
    public Tensor mean(Tensor a) {
        return wrap(cpuFallback.mean(a));
    }

    @Override
    public Tensor abs(Tensor a) {
        return wrap(cpuFallback.abs(a));
    }

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        return wrap(cpuFallback.crossEntropy(pred, target));
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        return wrap(cpuFallback.binaryCrossEntropy(pred, target));
    }

    @Override
    public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
        return wrap(cpuFallback.cast(a, dtype));
    }

    @Override
    public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
        if (device == DeviceType.METAL || device == DeviceType.CPU) {
            return a;
        }
        return wrap(cpuFallback.to(a, device));
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
        return wrap(cpuFallback.sqrt(a));
    }

    @Override
    public Tensor relu(Tensor a) {
        return wrap(cpuFallback.relu(a));
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        return wrap(cpuFallback.sigmoid(a));
    }

    @Override
    public Tensor tanh(Tensor a) {
        return wrap(cpuFallback.tanh(a));
    }

    @Override
    public Tensor log(Tensor a) {
        return wrap(cpuFallback.log(a));
    }

    @Override
    public Tensor exp(Tensor a) {
        return wrap(cpuFallback.exp(a));
    }

    @Override
    public Tensor silu(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return wrap(cpuFallback.silu(a));

        Shape shape = a.shape();
        int n = (int) shape.numel();
        HostAccessibleStorage bufferOut = allocateHostStorage(n * 4);

        return dispatchOutput(
                bufferOut,
                () -> metalBinding.silu(bufferOut.segment(), segment(a), n),
                () -> wrap(cpuFallback.silu(a)),
                output -> new DefaultTensor(shape, a.dtype(), a.device(), output, this));
    }

    @Override
    public Tensor flatten(Tensor a) {
        return wrap(cpuFallback.flatten(a));
    }

    @Override
    public Tensor unsqueeze(Tensor a, int dim) {
        return wrap(cpuFallback.unsqueeze(a, dim));
    }

    @Override
    public Tensor squeeze(Tensor a) {
        return wrap(cpuFallback.squeeze(a));
    }

    @Override
    public Tensor transpose(Tensor a) {
        return wrap(cpuFallback.transpose(a));
    }

    @Override
    public Tensor transpose(Tensor a, int d0, int d1) {
        return wrap(cpuFallback.transpose(a, d0, d1));
    }

    @Override
    public Tensor gelu(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return wrap(cpuFallback.gelu(a));

        Shape shape = a.shape();
        int n = (int) shape.numel();
        HostAccessibleStorage bufferOut = allocateHostStorage(n * 4);

        return dispatchOutput(
                bufferOut,
                () -> metalBinding.gelu(bufferOut.segment(), segment(a), n),
                () -> wrap(cpuFallback.gelu(a)),
                output -> new DefaultTensor(shape, a.dtype(), a.device(), output, this));
    }

    @Override
    public Tensor softmax(Tensor a, int dim) {
        if (!isNative || a.dtype() != DType.F32)
            return wrap(cpuFallback.softmax(a, dim));

        if (dim == a.shape().rank() - 1) {
            int rows = 1;
            for (int i = 0; i < dim; i++) {
                rows *= a.shape().dim(i);
            }
            int cols = (int) a.shape().dim(dim);

                HostAccessibleStorage bufferOut = allocateHostStorage(rows * cols * 4);
            return dispatchOutput(
                    bufferOut,
                    () -> metalBinding.softmaxRows(bufferOut.segment(), segment(a), rows, cols),
                    () -> wrap(cpuFallback.softmax(a, dim)),
                    output -> new DefaultTensor(a.shape(), a.dtype(), a.device(), output, this));
        }
        return wrap(cpuFallback.softmax(a, dim));
    }

    @Override
    public Tensor logSoftmax(Tensor a, int dim) {
        return wrap(cpuFallback.logSoftmax(a, dim));
    }

    @Override
    public Tensor mean(Tensor a, int dim, boolean keepDim) {
        return wrap(cpuFallback.mean(a, dim, keepDim));
    }

    @Override
    public Tensor sum(Tensor a) {
        return wrap(cpuFallback.sum(a));
    }

    @Override
    public Tensor sum(Tensor a, int dim, boolean keepDim) {
        return wrap(cpuFallback.sum(a, dim, keepDim));
    }

    @Override
    public Tensor max(Tensor a) {
        return wrap(cpuFallback.max(a));
    }

    @Override
    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        if (!isNative || input.dtype() != DType.F32)
            return wrap(cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps));

        Shape shape = input.shape();
        long sizeBytes = shape.numel() * byteSize(input.dtype());
        HostAccessibleStorage bufferOut = allocateHostStorage(sizeBytes);

        int n = 1;
        for (long s : normalizedShape) {
            n *= s;
        }
        int rows = (int) (shape.numel() / n);

        return dispatchOutput(
                bufferOut,
                () -> rows == 1
                        ? metalBinding.layerNorm(bufferOut.segment(), segment(input),
                                optionalSegment(weight), optionalSegment(bias), n, eps)
                        : metalBinding.layerNormRows(bufferOut.segment(), segment(input),
                                optionalSegment(weight), optionalSegment(bias), rows, n, eps),
                () -> wrap(cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps)),
                output -> new DefaultTensor(shape, input.dtype(), input.device(), output, this));
    }

    @Override
    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        if (!isNative || input.dtype() != DType.F32)
            return wrap(cpuFallback.rmsNorm(input, weight, eps));

        Shape shape = input.shape();
        int n = (int) shape.dim(shape.rank() - 1);
        int rows = 1;
        for (int i = 0; i < shape.rank() - 1; i++) {
            rows *= shape.dim(i);
        }

        HostAccessibleStorage bufferOut = allocateHostStorage(rows * n * 4);
        return dispatchOutput(
                bufferOut,
                () -> rows == 1
                        ? metalBinding.rmsNorm(bufferOut.segment(), segment(input), segment(weight), n, eps, false)
                        : metalBinding.rmsNormRows(bufferOut.segment(), segment(input),
                                segment(weight), rows, n, eps, false),
                () -> wrap(cpuFallback.rmsNorm(input, weight, eps)),
                output -> new DefaultTensor(shape, input.dtype(), input.device(), output, this));
    }

    @Override
    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        return wrap(cpuFallback.batchNorm(input, weight, bias, runningMean, runningVar, training, momentum, eps));
    }

    @Override
    public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return wrap(cpuFallback.conv2d(input, weight, bias, stride, padding, dilation, groups));
    }

    @Override
    public Tensor upsampleNearest2d(Tensor input, int outputH, int outputW) {
        return wrap(cpuFallback.upsampleNearest2d(input, outputH, outputW));
    }

    @Override
    public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        return wrap(cpuFallback.maxPool2d(input, kernelSize, stride, padding));
    }

    @Override
    public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        return wrap(cpuFallback.adaptiveAvgPool2d(input, outputH, outputW));
    }

    @Override
    public Tensor dropout(Tensor input, float p, boolean training) {
        return wrap(cpuFallback.dropout(input, p, training));
    }

    @Override
    public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
        return wrap(cpuFallback.embedding(weight, input, paddingIdx));
    }

    @Override
    public Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        if (!isNative || input.dtype() != DType.F32) {
            return wrap(cpuFallback.applyRoPE(input, posOffset, freqBase, isNeox));
        }

        int n = Math.toIntExact(input.numel());
        int headDim = Math.toIntExact(input.shape().dim(input.shape().rank() - 1)); // Last dimension is headDim

        HostAccessibleStorage bufferOut = allocateHostStorage((long) n * 4);
        return dispatchOutput(
                bufferOut,
                () -> metalBinding.rope(bufferOut.segment(), segment(input), n, headDim, posOffset, freqBase,
                        isNeox),
                () -> wrap(cpuFallback.applyRoPE(input, posOffset, freqBase, isNeox)),
                output -> new DefaultTensor(input.shape(), DType.F32, DeviceType.METAL, output, this));
    }


    @Override
    public Tensor cat(List<Tensor> tensors, int axis) {
        return wrap(cpuFallback.cat(tensors, axis));
    }

    @Override
    public long numel(Tensor a) {
        return cpuFallback.numel(a);
    }
}
