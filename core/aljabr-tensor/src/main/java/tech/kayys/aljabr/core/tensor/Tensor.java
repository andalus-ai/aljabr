package tech.kayys.aljabr.core.tensor;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.TensorStorage;

/**
 * The core abstraction of the Aljabr framework representing a multi-dimensional
 * array of elements.
 *
 * <p>A {@code Tensor} is immutable with respect to its mathematical properties
 * (shape, data type). By default, computation operations return a new
 * {@code Tensor} instance unless an explicitly mutable operation is used.</p>
 *
 * <p>Storage access is an optional capability for tensor implementations.
 * Eager tensors expose their backing storage; lazy or remote implementations
 * may materialize or reject direct storage access.</p>
 */
public interface Tensor {
    /**
     * Evaluates and materializes this Tensor if it is a lazy computation graph
     * node. For eager tensors, this returns the tensor itself.
     */
    Tensor eval();

    Shape shape();

    DeviceType device();

    DType dtype();

    ComputeBackend backend();

    /**
     * Returns this tensor's backing storage when the implementation exposes it.
     *
     * <p>Implementations that do not expose backing storage should retain the
     * default unsupported behavior. Lazy implementations may evaluate before
     * returning storage.</p>
     *
     * @throws UnsupportedOperationException when backing storage is not exposed
     */
    default TensorStorage storage() {
        throw new UnsupportedOperationException(
                "Backing storage is not exposed by " + getClass().getName());
    }

    Tensor add(Tensor other);

    Tensor sub(Tensor other);

    Tensor mul(Tensor other);

    Tensor mul(float scalar);

    Tensor div(float scalar);

    Tensor matmul(Tensor other);

    Tensor reshape(long... newShape);

    Tensor softmax();

    Tensor slice(long[] offsets, long[] sizes);

    Tensor pow(float exponent);

    Tensor mean();

    Tensor abs();

    Tensor crossEntropy(Tensor target);

    Tensor binaryCrossEntropy(Tensor target);

    Tensor div(Tensor other);

    Tensor add(float scalar);

    Tensor zerosLike();

    Tensor sqrt();

    Tensor cast(DType dtype);

    Tensor to(DeviceType device);

    default Tensor toFP32() {
        return cast(DType.F32);
    }

    float item();

    void backward();

    Tensor grad();

    void setGrad(Tensor grad);

    boolean requiresGrad();

    void setRequiresGrad(boolean requiresGrad);

    default void release() {
    }

    static Tensor randn(long... shape) {
        return TensorFactory.randn(shape);
    }

    static Tensor randn(ComputeBackend backend, long... shape) {
        return TensorFactory.randn(backend, shape);
    }

    static Tensor zeros(long... shape) {
        return TensorFactory.zeros(shape);
    }

    static Tensor zeros(ComputeBackend backend, long... shape) {
        return TensorFactory.zeros(backend, shape);
    }

    static Tensor ones(long... shape) {
        return TensorFactory.ones(shape);
    }

    static Tensor ones(ComputeBackend backend, long... shape) {
        return TensorFactory.ones(backend, shape);
    }

    static Tensor full(float value, long... shape) {
        return TensorFactory.full(value, shape);
    }

    static Tensor full(float value, ComputeBackend backend, long... shape) {
        return TensorFactory.full(value, backend, shape);
    }

    static Tensor of(float[] data, long... shape) {
        return TensorFactory.of(data, shape);
    }

    static Tensor of(float[] data, ComputeBackend backend, long... shape) {
        return TensorFactory.of(data, backend, shape);
    }

    // Common operations
    Tensor relu();

    Tensor gelu();

    Tensor sigmoid();

    Tensor tanh();

    Tensor log();

    Tensor exp();

    Tensor silu();

    Tensor flatten();

    Tensor unsqueeze(int dim);

    Tensor squeeze();

    Tensor transpose();

    Tensor transpose(int dim0, int dim1);

    Tensor softmax(int dim);

    Tensor logSoftmax(int dim);

    Tensor mean(int dim, boolean keepDim);

    Tensor sum();

    Tensor sum(int dim, boolean keepDim);

    java.util.List<Tensor> split(int axis, int parts);

    Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps);

    Tensor rmsNorm(Tensor weight, float eps);

    Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training,
            float momentum, float eps);

    Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups);

    Tensor maxPool2d(int kernelSize, int stride, int padding);

    Tensor adaptiveAvgPool2d(int outputH, int outputW);

    Tensor dropout(float p, boolean training);

    Tensor attention(Tensor K, Tensor V);

    Tensor embedding(Tensor weight, long paddingIdx);

    default Tensor applyRoPE(int posOffset, float freqBase, boolean isNeox) {
        throw new UnsupportedOperationException("RoPE is not supported by " + getClass().getSimpleName());
    }

    default Tensor updateKVCache(int layerIdx, String type) {
        throw new UnsupportedOperationException("KVCache not supported by " + getClass().getSimpleName());
    }

    long numel();

    default float[] toFloatArray() {
        throw new UnsupportedOperationException("Float-array export is not supported by " + getClass().getSimpleName());
    }
}
