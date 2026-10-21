package tech.kayys.aljabr.backend.metal;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.metal.binding.MetalBinding;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Metal hardware-accelerated computation backend.
 */
public class MetalComputeBackend implements ComputeBackend {

    private Tensor wrap(Tensor cpuRes) {
        if (cpuRes instanceof tech.kayys.aljabr.core.tensor.DefaultTensor dt) {
            return new tech.kayys.aljabr.core.tensor.DefaultTensor(dt.shape(), dt.dtype(), dt.device(),
                    dt.buffer(), this);
        }
        return cpuRes;
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

    private DefaultTensor asDefault(Tensor t) {
        if (t instanceof DefaultTensor dt) {
            return dt;
        }
        throw new IllegalArgumentException("MetalComputeBackend only supports DefaultTensor");
    }

    private long byteSize(DType dtype) {
        return switch (dtype) {
            case F32, I32 -> 4;
            case F16, BF16 -> 2;
            case I8, INT8, Q8_0 -> 1;
            case Q4_K, Q4_0 -> 0;
        };
    }

    private CpuBuffer allocate(long sizeBytes) {
        return new CpuBuffer(sizeBytes);
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

        DefaultTensor da = asDefault(a);
        DefaultTensor db = asDefault(b);

        if (b.dtype() == tech.kayys.aljabr.core.tensor.DType.Q4_K
                || b.dtype() == tech.kayys.aljabr.core.tensor.DType.Q8_0) {
            return matmulQuantized(da, db);
        }

        int M = (int) a.shape().dim(a.shape().rank() - 2);
        int K = (int) a.shape().dim(a.shape().rank() - 1);
        int N = (int) b.shape().dim(b.shape().rank() - 1);

        Shape shapeC = new Shape(M, N);
        long sizeBytes = shapeC.numel() * byteSize(a.dtype());

        CpuBuffer bufferC = allocate(sizeBytes);
        int status = metalBinding.matmul(bufferC.segment(), da.buffer().segment(), db.buffer().segment(), M, K, N, 1.0f,
                0.0f);
        if (status != 0) {
            System.err.println("Metal matmul falling back to CPU for a=" + a.dtype() + " (shape=" + a.shape() + "), b="
                    + b.dtype() + " (shape=" + b.shape() + "), M=" + M + ", K=" + K + ", N=" + N);
            return wrap(cpuFallback.matmul(a, b));
        }

        return new DefaultTensor(shapeC, a.dtype(), a.device(), bufferC, this);
    }

    private Tensor matmulQuantized(DefaultTensor a, DefaultTensor db) {
        int M = (int) a.shape().dim(a.shape().rank() - 2);
        int K = (int) a.shape().dim(a.shape().rank() - 1);
        int N = (int) db.shape().dim(0); // weight rows = output dim

        Shape shapeC = new Shape(M, N);
        CpuBuffer bufferC = allocate((long) M * N * 4);

        int status = 0;
        if (db.dtype() == tech.kayys.aljabr.core.tensor.DType.Q4_K) {
            for (int m = 0; m < M; m++) {
                status = metalBinding.matvecTransposedRightQ4K(
                        bufferC.segment().asSlice(m * N * 4L, N * 4L),
                        a.buffer().segment().asSlice(m * K * 4L, K * 4L),
                        db.buffer().segment(), K, N);
                if (status != 0)
                    break;
            }
        } else if (db.dtype() == tech.kayys.aljabr.core.tensor.DType.Q8_0) {
            for (int m = 0; m < M; m++) {
                status = metalBinding.matvecTransposedRightQ8_0(
                        bufferC.segment().asSlice(m * N * 4L, N * 4L),
                        a.buffer().segment().asSlice(m * K * 4L, K * 4L),
                        db.buffer().segment(), K, N);
                if (status != 0)
                    break;
            }
        } else {
            status = -1;
        }

        if (status != 0) {
            System.err.println("Metal matmul falling back to CPU for a=" + a.dtype() + " (shape=" + a.shape() + "), b="
                    + db.dtype() + " (shape=" + db.shape() + "), M=" + M + ", K=" + K + ", N=" + N + ", status="
                    + status);
            return wrap(cpuFallback.matmul(a, db));
        }

        return new DefaultTensor(shapeC, tech.kayys.aljabr.core.tensor.DType.F32, a.device(), bufferC, this);
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return wrap(cpuFallback.reshape(a, newShape));
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        if (!isNative)
            return wrap(cpuFallback.attention(Q, K, V));

        DefaultTensor dQ = asDefault(Q);
        DefaultTensor dK = asDefault(K);
        DefaultTensor dV = asDefault(V);

        int B = (int) Q.shape().dim(0);
        int T = (int) Q.shape().dim(1);
        int H = (int) Q.shape().dim(2);
        int Hkv = (int) K.shape().dim(2);
        int D = (int) Q.shape().dim(3);
        int Skv = (int) K.shape().dim(1);

        Shape shapeOut = Q.shape();
        long sizeBytes = shapeOut.numel() * byteSize(Q.dtype());
        CpuBuffer bufferOut = allocate(sizeBytes);

        int status;
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
            java.lang.foreign.MemorySegment contextLens = arena.allocate(java.lang.foreign.ValueLayout.JAVA_INT, B);
            for (int b = 0; b < B; b++) {
                contextLens.setAtIndex(java.lang.foreign.ValueLayout.JAVA_INT, b, Skv);
            }

            java.lang.foreign.MemorySegment empty = java.lang.foreign.MemorySegment.NULL;

            if (H == Hkv) {
                status = metalBinding.flashAttention(bufferOut.segment(), dQ.buffer().segment(), dK.buffer().segment(),
                        dV.buffer().segment(),
                        B, T, H, D);
            } else {
                status = metalBinding.attentionGqa(bufferOut.segment(), dQ.buffer().segment(), dK.buffer().segment(),
                        dV.buffer().segment(),
                        empty, contextLens, B, T, H, Hkv, D, 16, 1024, (float) (1.0 / Math.sqrt(D)), 1, 0.0f);
            }
        }

        if (status != 0) {
            return wrap(cpuFallback.attention(Q, K, V));
        }

        return new DefaultTensor(shapeOut, Q.dtype(), Q.device(), bufferOut, this);
    }

    @Override
    public Tensor softmax(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return wrap(cpuFallback.softmax(a));

        DefaultTensor da = asDefault(a);
        Shape shape = a.shape();
        int n = (int) shape.numel();
        CpuBuffer bufferOut = allocate(n * 4);

        int status = metalBinding.softmax(bufferOut.segment(), da.buffer().segment(), n);
        if (status != 0) {
            return wrap(cpuFallback.softmax(a));
        }

        return new DefaultTensor(shape, a.dtype(), a.device(), bufferOut, this);
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
        CpuBuffer buffer = allocate(sizeBytes);
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

        DefaultTensor da = asDefault(a);
        Shape shape = a.shape();
        int n = (int) shape.numel();
        CpuBuffer bufferOut = allocate(n * 4);

        int status = metalBinding.silu(bufferOut.segment(), da.buffer().segment(), n);
        if (status != 0) {
            return wrap(cpuFallback.silu(a));
        }

        return new DefaultTensor(shape, a.dtype(), a.device(), bufferOut, this);
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

        DefaultTensor da = asDefault(a);
        Shape shape = a.shape();
        int n = (int) shape.numel();
        CpuBuffer bufferOut = allocate(n * 4);

        int status = metalBinding.gelu(bufferOut.segment(), da.buffer().segment(), n);
        if (status != 0) {
            return wrap(cpuFallback.gelu(a));
        }

        return new DefaultTensor(shape, a.dtype(), a.device(), bufferOut, this);
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

            DefaultTensor da = asDefault(a);
            CpuBuffer bufferOut = allocate(rows * cols * 4);
            int status = metalBinding.softmaxRows(bufferOut.segment(), da.buffer().segment(), rows, cols);
            if (status == 0) {
                return new DefaultTensor(a.shape(), a.dtype(), a.device(), bufferOut, this);
            }
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

        DefaultTensor dInput = asDefault(input);
        DefaultTensor dWeight = weight != null ? asDefault(weight) : null;
        DefaultTensor dBias = bias != null ? asDefault(bias) : null;

        Shape shape = input.shape();
        long sizeBytes = shape.numel() * byteSize(input.dtype());
        CpuBuffer bufferOut = allocate(sizeBytes);

        int n = 1;
        for (long s : normalizedShape) {
            n *= s;
        }
        int rows = (int) (shape.numel() / n);

        int status;
        if (rows == 1) {
            status = metalBinding.layerNorm(bufferOut.segment(), dInput.buffer().segment(),
                    dWeight != null ? dWeight.buffer().segment() : MemorySegment.NULL,
                    dBias != null ? dBias.buffer().segment() : MemorySegment.NULL,
                    n, eps);
        } else {
            status = metalBinding.layerNormRows(bufferOut.segment(), dInput.buffer().segment(),
                    dWeight != null ? dWeight.buffer().segment() : MemorySegment.NULL,
                    dBias != null ? dBias.buffer().segment() : MemorySegment.NULL,
                    rows, n, eps);
        }

        if (status != 0) {
            return wrap(cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps));
        }

        return new DefaultTensor(shape, input.dtype(), input.device(), bufferOut, this);
    }

    @Override
    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        if (!isNative || input.dtype() != DType.F32)
            return wrap(cpuFallback.rmsNorm(input, weight, eps));

        DefaultTensor dInput = asDefault(input);
        DefaultTensor dWeight = asDefault(weight);

        Shape shape = input.shape();
        int n = (int) shape.dim(shape.rank() - 1);
        int rows = 1;
        for (int i = 0; i < shape.rank() - 1; i++) {
            rows *= shape.dim(i);
        }

        CpuBuffer bufferOut = allocate(rows * n * 4);
        int status;
        if (rows == 1) {
            status = metalBinding.rmsNorm(bufferOut.segment(), dInput.buffer().segment(), dWeight.buffer().segment(), n,
                    eps, false);
        } else {
            status = metalBinding.rmsNormRows(bufferOut.segment(), dInput.buffer().segment(),
                    dWeight.buffer().segment(), rows, n, eps, false);
        }

        if (status != 0) {
            return wrap(cpuFallback.rmsNorm(input, weight, eps));
        }

        return new DefaultTensor(shape, input.dtype(), input.device(), bufferOut, this);
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

        DefaultTensor dInput = (DefaultTensor) input;
        int n = (int) dInput.shape().numel();
        int headDim = (int) dInput.shape().dim(dInput.shape().rank() - 1); // Last dimension is headDim

        CpuBuffer bufferOut = allocate((long) n * 4);
        int status = metalBinding.rope(bufferOut.segment(), dInput.buffer().segment(), n, headDim, posOffset, freqBase,
                isNeox);

        if (status != 0) {
            bufferOut.release();
            return wrap(cpuFallback.applyRoPE(input, posOffset, freqBase, isNeox));
        }
        return new DefaultTensor(input.shape(), DType.F32, DeviceType.METAL, bufferOut, this);
    }

    @Override
    public long numel(Tensor a) {
        return cpuFallback.numel(a);
    }
}
