package tech.kayys.aljabr.backend.hat;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.List;
import hat.Accelerator.Compute;
import jdk.incubator.code.Reflect;

public class HatComputeBackend implements ComputeBackend {
    private static final Logger LOG = Logger.getLogger(HatComputeBackend.class);
    private final CpuBackend cpuFallback;

    public HatComputeBackend() {
        LOG.info("Initializing HAT Compute Backend...");
        this.cpuFallback = new CpuBackend();
        LOG.info("HAT Compute Backend initialized successfully with CPU fallback.");
    }

    private UnsupportedOperationException unimplemented(String op) {
        return new UnsupportedOperationException("HAT backend does not yet implement: " + op);
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
    public Tensor mul(Tensor a, Tensor b) {
        return cpuFallback.mul(a, b);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        // We ensure a and b are F32 tensors.
        // In a real implementation we would dynamically route based on dtype.
        int size = (int) a.shape().dims()[0]; // Assuming square matrix for naive example

        // 1. Initialize HAT Accelerator (uses default backend, e.g., CUDA or OpenCL)
        hat.Accelerator accelerator = new hat.Accelerator(java.lang.invoke.MethodHandles.lookup());

        // 2. Wrap the MemorySegments or float arrays into HAT F32Arrays
        hat.buffer.F32Array hatA = hat.buffer.F32Array.create(accelerator, (int) a.shape().numel());
        hatA.copyFrom(a.toFloatArray());

        hat.buffer.F32Array hatB = hat.buffer.F32Array.create(accelerator, (int) b.shape().numel());
        hatB.copyFrom(b.toFloatArray());

        hat.buffer.F32Array hatC = hat.buffer.F32Array.create(accelerator, (int) a.shape().numel());

        // 3. Dispatch the compute graph with our custom CodeReflection Kernel
        accelerator.compute((@Reflect Compute) (cc) -> HatMatmulKernel.mxmF32Dispatch(cc, hatA, hatB, hatC, size));

        // 4. Map the result back to Aljabr's off-heap tensor representation
        float[] resultArr = new float[(int) a.shape().numel()];
        hatC.copyTo(resultArr);

        // Use cpuFallback temporarily to construct the final Tensor object properly
        // from array
        // (A real production implementation would construct a native Aljabr Tensor
        // directly over the HAT buffer)
        Tensor res = cpuFallback.matmul(a, b);
        System.arraycopy(resultArr, 0, res.toFloatArray(), 0, resultArr.length);

        return res;
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        return cpuFallback.div(a, b);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        return cpuFallback.mul(a, scalar);
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return cpuFallback.div(a, scalar);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        return cpuFallback.addScalar(a, scalar);
    }

    @Override
    public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
        return cpuFallback.embedding(weight, input, paddingIdx);
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return cpuFallback.reshape(a, newShape);
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
    public Tensor transpose(Tensor a, int dim0, int dim1) {
        return cpuFallback.transpose(a, dim0, dim1);
    }

    @Override
    public Tensor softmax(Tensor a) {
        return cpuFallback.softmax(a);
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
    public Tensor mean(Tensor a) {
        return cpuFallback.mean(a);
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
    public Tensor abs(Tensor a) {
        return cpuFallback.abs(a);
    }

    @Override
    public Tensor relu(Tensor a) {
        return cpuFallback.relu(a);
    }

    @Override
    public Tensor gelu(Tensor a) {
        return cpuFallback.gelu(a);
    }

    @Override
    public Tensor silu(Tensor a) {
        return cpuFallback.silu(a);
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
    public Tensor sqrt(Tensor a) {
        return cpuFallback.sqrt(a);
    }

    @Override
    public Tensor exp(Tensor a) {
        return cpuFallback.exp(a);
    }

    @Override
    public Tensor log(Tensor a) {
        return cpuFallback.log(a);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        return cpuFallback.pow(a, exponent);
    }

    @Override
    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        return cpuFallback.layerNorm(input, normalizedShape, weight, bias, eps);
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
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        return cpuFallback.crossEntropy(pred, target);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        return cpuFallback.binaryCrossEntropy(pred, target);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        return cpuFallback.attention(Q, K, V);
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        return cpuFallback.zerosLike(a);
    }

    @Override
    public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
        return cpuFallback.cast(a, dtype);
    }

    @Override
    public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
        return cpuFallback.to(a, device);
    }

    @Override
    public long numel(Tensor a) {
        return cpuFallback.numel(a);
    }

    @Override
    public Tensor applyRoPE(Tensor input, int position, float theta, boolean isNeox) {
        return cpuFallback.applyRoPE(input, position, theta, isNeox);
    }
}