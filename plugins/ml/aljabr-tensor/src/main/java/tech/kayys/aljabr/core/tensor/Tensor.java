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
public interface Tensor extends AutoCloseable {
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

    /**
     * Concatenates tensors along an existing dimension.
     *
     * <p>All inputs must use the same backend and have compatible non-concatenated
     * dimensions. The first tensor determines the backend.</p>
     */
    static Tensor cat(java.util.List<Tensor> tensors, int axis) {
        return DefaultTensor.cat(tensors, axis);
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

    /**
     * Starts reverse-mode differentiation from this tensor.
     *
     * <p>Tensor implementations that do not provide an autograd engine must
     * fail explicitly when gradients are required; silently accepting the
     * call would make training appear to succeed without computing gradients.
     * Calls on tensors that do not require gradients may be no-ops.</p>
     *
     * @throws UnsupportedOperationException when gradients are required but
     *         this tensor implementation has no autograd support
     */
    void backward();

    /**
     * Starts reverse-mode differentiation with an explicit output gradient.
     * This is required for non-scalar outputs. The supplied gradient is
     * borrowed and must have the same shape as this tensor.
     *
     * @param outputGradient incoming gradient for this tensor
     * @throws UnsupportedOperationException when this implementation does not
     *         provide reverse-mode differentiation
     */
    default void backward(Tensor outputGradient) {
        throw new UnsupportedOperationException(
                "Explicit-gradient backward is not supported by " + getClass().getName());
    }

    /**
     * Returns the gradient currently owned by this tensor, or {@code null}.
     * The returned reference is borrowed; callers must not release it.
     */
    Tensor grad();

    /**
     * Assigns the gradient owned by this tensor.
     *
     * <p>Ownership of {@code grad} is transferred to this tensor. Replacing an
     * existing gradient releases the previously owned gradient. Passing
     * {@code null} clears the gradient.</p>
     *
     * @param grad gradient tensor to take ownership of, or {@code null}
     */
    void setGrad(Tensor grad);

    boolean requiresGrad();

    void setRequiresGrad(boolean requiresGrad);

    /**
     * Releases this tensor's owned resources. Implementations that own storage
     * must make repeated release calls harmless.
     */
    default void release() {
    }

    /**
     * Releases this tensor and its owned resources.
     * Views and derived tensors retain their own storage references and may
     * outlive the tensor from which they were created.
     */
    @Override
    default void close() {
        release();
    }

    /** @deprecated Use {@link #randn(ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    static Tensor randn(long... shape) {
        return TensorFactory.randn(shape);
    }

    static Tensor randn(ComputeBackend backend, long... shape) {
        return TensorFactory.randn(backend, shape);
    }

    /** @deprecated Use {@link #zeros(ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    static Tensor zeros(long... shape) {
        return TensorFactory.zeros(shape);
    }

    static Tensor zeros(ComputeBackend backend, long... shape) {
        return TensorFactory.zeros(backend, shape);
    }

    /** @deprecated Use {@link #ones(ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    static Tensor ones(long... shape) {
        return TensorFactory.ones(shape);
    }

    static Tensor ones(ComputeBackend backend, long... shape) {
        return TensorFactory.ones(backend, shape);
    }

    /** @deprecated Use {@link #full(float, ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    static Tensor full(float value, long... shape) {
        return TensorFactory.full(value, shape);
    }

    static Tensor full(float value, ComputeBackend backend, long... shape) {
        return TensorFactory.full(value, backend, shape);
    }

    /** @deprecated Use {@link #of(float[], ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
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

    /**
     * Resizes an NCHW tensor using nearest-neighbor sampling.
     *
     * @param outputH positive output height
     * @param outputW positive output width
     */
    default Tensor upsampleNearest2d(int outputH, int outputW) {
        return backend().upsampleNearest2d(this, outputH, outputW);
    }

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
