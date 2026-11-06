package tech.kayys.aljabr.backend.hat;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import java.util.List;
import hat.Accelerator.Compute;
import jdk.incubator.code.Reflect;

public class HatComputeBackend implements ComputeBackend {
    private static final Logger LOG = Logger.getLogger(HatComputeBackend.class);
    private final CpuBackend cpuFallback;

    /*
     * HAT's Java multi-threaded backend reuses mutable dispatch state, and
     * Accelerator caches ComputeContext instances in a non-concurrent map.
     * Keep one accelerator per Aljabr backend, but serialize access until HAT
     * exposes a verified concurrent-dispatch contract.
     */
    private final Object hatExecutionLock = new Object();
    private hat.Accelerator accelerator;

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
        java.util.Objects.requireNonNull(a, "left matrix must not be null");
        java.util.Objects.requireNonNull(b, "right matrix must not be null");

        if (a.shape().rank() != 2 || b.shape().rank() != 2) {
            throw new IllegalArgumentException("HAT matmul currently supports rank-2 matrices only");
        }
        if (a.dtype() != tech.kayys.aljabr.core.tensor.DType.F32
                || b.dtype() != tech.kayys.aljabr.core.tensor.DType.F32) {
            throw new IllegalArgumentException("HAT matmul currently supports F32 tensors only");
        }

        long[] aShape = a.shape().dims();
        long[] bShape = b.shape().dims();
        int rows = Math.toIntExact(aShape[0]);
        int reductionSize = Math.toIntExact(aShape[1]);
        int rightReductionSize = Math.toIntExact(bShape[0]);
        int columns = Math.toIntExact(bShape[1]);
        if (reductionSize != rightReductionSize) {
            throw new IllegalArgumentException("Matrix dimension mismatch: "
                    + reductionSize + " != " + rightReductionSize);
        }

        int outputLength = Math.multiplyExact(rows, columns);
        float[] result = new float[outputLength];
        if (outputLength == 0 || reductionSize == 0) {
            return TensorFactory.of(result, rows, columns);
        }

        synchronized (hatExecutionLock) {
            if (accelerator == null) {
                accelerator = new hat.Accelerator(java.lang.invoke.MethodHandles.lookup());
            }

            hat.buffer.F32Array hatA = hat.buffer.F32Array.create(
                    accelerator, Math.toIntExact(a.numel()));
            hat.buffer.F32Array hatB = hat.buffer.F32Array.create(
                    accelerator, Math.toIntExact(b.numel()));
            hat.buffer.F32Array hatC = hat.buffer.F32Array.create(accelerator, outputLength);

            hatA.copyFrom(a.toFloatArray());
            hatB.copyFrom(b.toFloatArray());

            accelerator.compute((@Reflect Compute) cc ->
                    HatMatmulKernel.mxmF32Dispatch(
                            cc, hatA, hatB, hatC, rows, columns, reductionSize));
            hatC.copyTo(result);
        }

        // Tensor.toFloatArray() returns a copy. Construct the result from the
        // computed HAT values instead of attempting to mutate that exported copy.
        return TensorFactory.of(result, rows, columns);
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