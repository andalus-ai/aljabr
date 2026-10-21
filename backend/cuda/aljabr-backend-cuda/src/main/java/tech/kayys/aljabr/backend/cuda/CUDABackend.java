package tech.kayys.aljabr.backend.cuda;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.cuda.binding.CudaBinding;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * CUDA hardware-accelerated computation backend.
 */
public class CUDABackend implements ComputeBackend {

    private static final Logger LOG = Logger.getLogger(CUDABackend.class);
    private static final String FORCE_CPU_PROPERTY = "aljabr.kernel.force.cpu";

    private final CudaBinding cudaBinding;
    private final CpuBackend cpuFallback;
    private final boolean isNative;
    private final boolean forceCpu;

    private MemorySegment cublasHandle = MemorySegment.NULL;

    public CUDABackend() {
        this.forceCpu = Boolean.parseBoolean(System.getProperty(FORCE_CPU_PROPERTY, "false"));

        if (forceCpu) {
            LOG.info("CUDABackend forced into CPU mode by system properties.");
            CudaBinding.initializeFallback();
            this.cudaBinding = CudaBinding.getInstance();
            this.isNative = false;
            this.cpuFallback = new CpuBackend();
            return;
        }

        boolean loaded = CudaBinding.initialize();
        if (!loaded) {
            LOG.warn("Failed to initialize CudaBinding. CUDABackend will operate in CPU fallback mode.");
            CudaBinding.initializeFallback();
        }

        this.cudaBinding = CudaBinding.getInstance();
        this.isNative = cudaBinding.isNativeAvailable();
        this.cpuFallback = new CpuBackend();

        if (this.isNative) {
            this.cublasHandle = cudaBinding.cublasCreate();
            LOG.infof("Initialized CUDABackend [Device: %s]", cudaBinding.deviceName(0));
        }
    }

    private DefaultTensor asDefault(Tensor t) {
        if (t instanceof DefaultTensor dt) {
            return dt;
        }
        throw new IllegalArgumentException("CUDABackend only supports DefaultTensor");
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
        return new CpuBuffer(sizeBytes); // Using CpuBuffer assuming managed memory or host allocation initially
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        return cpuFallback.add(a, b);
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        return cpuFallback.sub(a, b);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        return cpuFallback.mul(a, scalar);
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        return cpuFallback.mul(a, b);
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return cpuFallback.div(a, scalar);
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        return cpuFallback.div(a, b);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        return cpuFallback.addScalar(a, scalar);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        if (!isNative || cublasHandle.equals(MemorySegment.NULL) || a.dtype() != DType.F32)
            return cpuFallback.matmul(a, b);

        DefaultTensor da = asDefault(a);
        DefaultTensor db = asDefault(b);

        int M = (int) a.shape().dim(a.shape().rank() - 2);
        int K = (int) a.shape().dim(a.shape().rank() - 1);
        int N = (int) b.shape().dim(b.shape().rank() - 1);

        Shape shapeC = new Shape(M, N);
        long sizeBytesA = a.numel() * byteSize(a.dtype());
        long sizeBytesB = b.numel() * byteSize(b.dtype());
        long sizeBytesC = shapeC.numel() * byteSize(a.dtype());

        // 1. Allocate GPU memory
        MemorySegment d_A = cudaBinding.cudaMalloc(sizeBytesA);
        MemorySegment d_B = cudaBinding.cudaMalloc(sizeBytesB);
        MemorySegment d_C = cudaBinding.cudaMalloc(sizeBytesC);

        // 2. Copy host to device
        cudaBinding.cudaMemcpy(d_A, da.buffer().segment(), sizeBytesA, CudaBinding.cudaMemcpyHostToDevice);
        cudaBinding.cudaMemcpy(d_B, db.buffer().segment(), sizeBytesB, CudaBinding.cudaMemcpyHostToDevice);

        // 3. cuBLAS sgemm (Note: cuBLAS is column-major, so we compute B^T * A^T to get
        // row-major C)
        // cublasSgemm(handle, transb, transa, N, M, K, alpha, d_B, N, d_A, K, beta,
        // d_C, N)
        // Or simple N=false, T=false but swap A and B -> C^T = B^T * A^T
        int status = cudaBinding.cublasSgemm(cublasHandle,
                CudaBinding.CUBLAS_OP_N, CudaBinding.CUBLAS_OP_N,
                N, M, K,
                1.0f,
                d_B, N,
                d_A, K,
                0.0f,
                d_C, N);

        if (status != 0) {
            cudaBinding.cudaFree(d_A);
            cudaBinding.cudaFree(d_B);
            cudaBinding.cudaFree(d_C);
            return cpuFallback.matmul(a, b);
        }

        // 4. Copy device to host
        CpuBuffer bufferC = allocate(sizeBytesC);
        cudaBinding.cudaMemcpy(bufferC.segment(), d_C, sizeBytesC, CudaBinding.cudaMemcpyDeviceToHost);

        // 5. Free GPU memory
        cudaBinding.cudaFree(d_A);
        cudaBinding.cudaFree(d_B);
        cudaBinding.cudaFree(d_C);

        return new DefaultTensor(shapeC, a.dtype(), a.device(), bufferC, this);
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return cpuFallback.reshape(a, newShape);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        return cpuFallback.attention(Q, K, V);
    }

    @Override
    public Tensor softmax(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return cpuFallback.softmax(a);

        DefaultTensor da = asDefault(a);
        Shape shape = a.shape();
        int n = (int) shape.numel();
        long sizeBytes = n * byteSize(a.dtype());

        MemorySegment d_in = cudaBinding.cudaMalloc(sizeBytes);
        MemorySegment d_out = cudaBinding.cudaMalloc(sizeBytes);
        cudaBinding.cudaMemcpy(d_in, da.buffer().segment(), sizeBytes, CudaBinding.cudaMemcpyHostToDevice);

        int status = cudaBinding.softmax(d_out, d_in, 1, n);
        if (status != 0) {
            cudaBinding.cudaFree(d_in);
            cudaBinding.cudaFree(d_out);
            return cpuFallback.softmax(a);
        }

        CpuBuffer bufferOut = allocate(sizeBytes);
        cudaBinding.cudaMemcpy(bufferOut.segment(), d_out, sizeBytes, CudaBinding.cudaMemcpyDeviceToHost);
        cudaBinding.cudaFree(d_in);
        cudaBinding.cudaFree(d_out);

        return new DefaultTensor(shape, a.dtype(), a.device(), bufferOut, this);
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        return cpuFallback.slice(a, offsets, sizes);
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        return cpuFallback.split(a, axis, parts);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        return cpuFallback.pow(a, exponent);
    }

    @Override
    public Tensor mean(Tensor a) {
        return cpuFallback.mean(a);
    }

    @Override
    public Tensor abs(Tensor a) {
        return cpuFallback.abs(a);
    }

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        return cpuFallback.crossEntropy(pred, target);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        return cpuFallback.binaryCrossEntropy(pred, target);
    }

    @Override
    public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
        return cpuFallback.cast(a, dtype);
    }

    @Override
    public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
        if (device == DeviceType.CUDA || device == DeviceType.CPU) {
            return a;
        }
        return cpuFallback.to(a, device);
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        return cpuFallback.zerosLike(a);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        return cpuFallback.sqrt(a);
    }

    @Override
    public Tensor relu(Tensor a) {
        return cpuFallback.relu(a);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        return cpuFallback.sigmoid(a);
    }

    @Override
    public Tensor tanh(Tensor a) {
        return cpuFallback.tanh(a);
    }

    @Override
    public Tensor log(Tensor a) {
        return cpuFallback.log(a);
    }

    @Override
    public Tensor exp(Tensor a) {
        return cpuFallback.exp(a);
    }

    @Override
    public Tensor silu(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return cpuFallback.silu(a);

        DefaultTensor da = asDefault(a);
        Shape shape = a.shape();
        int n = (int) shape.numel();
        long sizeBytes = n * byteSize(a.dtype());

        MemorySegment d_in = cudaBinding.cudaMalloc(sizeBytes);
        MemorySegment d_out = cudaBinding.cudaMalloc(sizeBytes);
        cudaBinding.cudaMemcpy(d_in, da.buffer().segment(), sizeBytes, CudaBinding.cudaMemcpyHostToDevice);

        int status = cudaBinding.silu(d_out, d_in, n);
        if (status != 0) {
            cudaBinding.cudaFree(d_in);
            cudaBinding.cudaFree(d_out);
            return cpuFallback.silu(a);
        }

        CpuBuffer bufferOut = allocate(sizeBytes);
        cudaBinding.cudaMemcpy(bufferOut.segment(), d_out, sizeBytes, CudaBinding.cudaMemcpyDeviceToHost);
        cudaBinding.cudaFree(d_in);
        cudaBinding.cudaFree(d_out);

        return new DefaultTensor(shape, a.dtype(), a.device(), bufferOut, this);
    }

    @Override
    public Tensor flatten(Tensor a) {
        return cpuFallback.flatten(a);
    }

    @Override
    public Tensor unsqueeze(Tensor a, int dim) {
        return cpuFallback.unsqueeze(a, dim);
    }

    @Override
    public Tensor squeeze(Tensor a) {
        return cpuFallback.squeeze(a);
    }

    @Override
    public Tensor transpose(Tensor a) {
        return cpuFallback.transpose(a);
    }

    @Override
    public Tensor transpose(Tensor a, int d0, int d1) {
        return cpuFallback.transpose(a, d0, d1);
    }

    @Override
    public Tensor gelu(Tensor a) {
        if (!isNative || a.dtype() != DType.F32)
            return cpuFallback.gelu(a);

        DefaultTensor da = asDefault(a);
        Shape shape = a.shape();
        int n = (int) shape.numel();
        long sizeBytes = n * byteSize(a.dtype());

        MemorySegment d_in = cudaBinding.cudaMalloc(sizeBytes);
        MemorySegment d_out = cudaBinding.cudaMalloc(sizeBytes);
        cudaBinding.cudaMemcpy(d_in, da.buffer().segment(), sizeBytes, CudaBinding.cudaMemcpyHostToDevice);

        int status = cudaBinding.gelu(d_out, d_in, n);
        if (status != 0) {
            cudaBinding.cudaFree(d_in);
            cudaBinding.cudaFree(d_out);
            return cpuFallback.gelu(a);
        }

        CpuBuffer bufferOut = allocate(sizeBytes);
        cudaBinding.cudaMemcpy(bufferOut.segment(), d_out, sizeBytes, CudaBinding.cudaMemcpyDeviceToHost);
        cudaBinding.cudaFree(d_in);
        cudaBinding.cudaFree(d_out);

        return new DefaultTensor(shape, a.dtype(), a.device(), bufferOut, this);
    }

    @Override
    public Tensor softmax(Tensor a, int dim) {
        return cpuFallback.softmax(a, dim);
    }

    @Override
    public Tensor logSoftmax(Tensor a, int dim) {
        return cpuFallback.logSoftmax(a, dim);
    }

    @Override
    public Tensor mean(Tensor a, int dim, boolean keepDim) {
        return cpuFallback.mean(a, dim, keepDim);
    }

    @Override
    public Tensor sum(Tensor a) {
        return cpuFallback.sum(a);
    }

    @Override
    public Tensor sum(Tensor a, int dim, boolean keepDim) {
        return cpuFallback.sum(a, dim, keepDim);
    }

    @Override
    public Tensor max(Tensor a) {
        return cpuFallback.max(a);
    }

    @Override
    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        if (!isNative || input.dtype() != DType.F32)
            return cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps);

        DefaultTensor dInput = asDefault(input);
        DefaultTensor dWeight = weight != null ? asDefault(weight) : null;
        DefaultTensor dBias = bias != null ? asDefault(bias) : null;

        Shape shape = input.shape();
        long sizeBytes = shape.numel() * byteSize(input.dtype());

        int n = 1;
        for (long s : normalizedShape) {
            n *= s;
        }
        int rows = (int) (shape.numel() / n);

        MemorySegment d_in = cudaBinding.cudaMalloc(sizeBytes);
        MemorySegment d_out = cudaBinding.cudaMalloc(sizeBytes);
        MemorySegment d_weight = dWeight != null ? cudaBinding.cudaMalloc(n * 4) : MemorySegment.NULL;
        MemorySegment d_bias = dBias != null ? cudaBinding.cudaMalloc(n * 4) : MemorySegment.NULL;

        cudaBinding.cudaMemcpy(d_in, dInput.buffer().segment(), sizeBytes, CudaBinding.cudaMemcpyHostToDevice);
        if (d_weight != MemorySegment.NULL)
            cudaBinding.cudaMemcpy(d_weight, dWeight.buffer().segment(), n * 4, CudaBinding.cudaMemcpyHostToDevice);
        if (d_bias != MemorySegment.NULL)
            cudaBinding.cudaMemcpy(d_bias, dBias.buffer().segment(), n * 4, CudaBinding.cudaMemcpyHostToDevice);

        int status;
        if (rows == 1) {
            status = cudaBinding.layerNorm(d_out, d_in, d_weight, d_bias, n, eps);
        } else {
            status = cudaBinding.layerNormRows(d_out, d_in, d_weight, d_bias, rows, n, eps);
        }

        if (status != 0) {
            cudaBinding.cudaFree(d_in);
            cudaBinding.cudaFree(d_out);
            if (d_weight != MemorySegment.NULL)
                cudaBinding.cudaFree(d_weight);
            if (d_bias != MemorySegment.NULL)
                cudaBinding.cudaFree(d_bias);
            return cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps);
        }

        CpuBuffer bufferOut = allocate(sizeBytes);
        cudaBinding.cudaMemcpy(bufferOut.segment(), d_out, sizeBytes, CudaBinding.cudaMemcpyDeviceToHost);

        cudaBinding.cudaFree(d_in);
        cudaBinding.cudaFree(d_out);
        if (d_weight != MemorySegment.NULL)
            cudaBinding.cudaFree(d_weight);
        if (d_bias != MemorySegment.NULL)
            cudaBinding.cudaFree(d_bias);

        return new DefaultTensor(shape, input.dtype(), input.device(), bufferOut, this);
    }

    @Override
    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        return cpuFallback.rmsNorm(input, weight, eps);
    }

    @Override
    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        return cpuFallback.batchNorm(input, weight, bias, runningMean, runningVar, training, momentum, eps);
    }

    @Override
    public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return cpuFallback.conv2d(input, weight, bias, stride, padding, dilation, groups);
    }

    @Override
    public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        return cpuFallback.maxPool2d(input, kernelSize, stride, padding);
    }

    @Override
    public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        return cpuFallback.adaptiveAvgPool2d(input, outputH, outputW);
    }

    @Override
    public Tensor dropout(Tensor input, float p, boolean training) {
        return cpuFallback.dropout(input, p, training);
    }

    @Override
    public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
        return cpuFallback.embedding(weight, input, paddingIdx);
    }

    @Override
    public Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        return cpuFallback.applyRoPE(input, posOffset, freqBase, isNeox);
    }

    @Override
    public long numel(Tensor a) {
        return cpuFallback.numel(a);
    }
}