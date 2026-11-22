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
 *
 * <p>Ownership is explicit: methods documented as returning borrowed values do
 * not transfer ownership. A borrowed value must not be released by the caller.
 * To keep borrowed storage alive independently of its tensor, call
 * {@link TensorStorage#retain()} while the tensor is live and balance that
 * reference with {@link TensorStorage#release()}.</p>
 */
public interface Tensor extends AutoCloseable {
    /**
     * Evaluates and materializes this Tensor if it is a lazy computation graph
     * node. For eager tensors, this returns the tensor itself.
     *
     * <p>The returned tensor is borrowed from this tensor wrapper. This method
     * does not transfer ownership or create an independent reference; do not
     * release the returned tensor. Close the wrapper to release a materialized
     * result owned by a lazy implementation.</p>
     *
     * @return this tensor or its borrowed materialized result
     */
    Tensor eval();

    Shape shape();

    /**
     * Returns the tensor's immutable element-based layout.
     *
     * <p>Implementations with canonical contiguous storage may use this
     * default. Implementations with views, offsets, or non-canonical strides
     * must override it. The descriptor does not imply that every backend
     * operation supports arbitrary layouts.</p>
     */
    default TensorLayout layout() {
        return TensorLayout.contiguous(shape());
    }

    DeviceType device();

    DType dtype();

    ComputeBackend backend();

    /**
     * Returns this tensor's backing storage when the implementation exposes it.
     *
     * <p>The returned storage is borrowed: this method does not retain it or
     * transfer ownership. Its lifetime is therefore bounded by the tensor's
     * ownership of that storage. Call {@link TensorStorage#retain()} before
     * allowing the storage to outlive this tensor, and release that additional
     * reference when finished. Lazy implementations may evaluate before
     * returning storage; the returned storage remains borrowed from the
     * materialized result.</p>
     *
     * @return borrowed backing storage; never release this reference directly
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

    /**
     * Selects entries along one axis using a rank-one I32 index tensor.
     * Index order and duplicates are preserved; indices must be in
     * {@code [0, shape().dim(axis))}. Negative axes count from the end.
     *
     * <p>The initial CPU implementation supports F32 source tensors. The
     * operation is currently forward-only; differentiable use is rejected
     * while the scatter-add VJP is being implemented.</p>
     */
    Tensor indexSelect(int axis, Tensor indices);

    /** Adds source slices at rank-one I32 indices; repeated indices accumulate. */
    Tensor indexAdd(int axis, Tensor indices, Tensor source);


    /** Adds source slices at rank-one I32 indices; repeated indices accumulate. */
    Tensor indexAdd(int axis, Tensor indices, Tensor source);


    Tensor pow(float exponent);

    Tensor mean();

    /** Returns the element-wise sign (-1, 0, or 1; NaN is preserved). */
    default Tensor sign() {
        return backend().sign(this);
    }

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
     * The returned reference is borrowed; callers must not release it. If the
     * gradient must outlive this tensor, first establish an independent
     * ownership reference according to the gradient implementation's contract.
     */
    Tensor grad();

    /**
     * Assigns the gradient owned by this tensor.
     *
     * <p>Ownership of {@code grad} is transferred to this tensor. Replacing an
     * existing gradient releases the previously owned gradient. Passing
     * {@code null} clears the gradient. The caller relinquishes its ownership
     * when this method succeeds and must not subsequently release the
     * transferred reference. To retain a separate caller-owned reference,
     * retain the gradient before transferring it.</p>
     *
     * @param grad gradient tensor whose ownership is transferred, or {@code null}
     */
    void setGrad(Tensor grad);

    boolean requiresGrad();

    void setRequiresGrad(boolean requiresGrad);

    /**
     * Releases this tensor's owned resources, including storage and any
     * gradient whose ownership was transferred to it. Borrowed values obtained
     * from this tensor must not be released by callers. Implementations that
     * own storage must make repeated release calls harmless.
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

    /** Creates an I32 tensor from typed integer data using an explicit backend. */
    static Tensor of(int[] data, ComputeBackend backend, long... shape) {
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

    /** Swaps the final two dimensions. The tensor must have rank at least two. */
    Tensor transpose();

    /**
     * Swaps two dimensions. Negative axes count from the final dimension
     * (for example, {@code -1} is the final axis). Both axes must be in
     * {@code [-rank, rank)}; using the same axis is a valid no-op.
     */
    Tensor transpose(int dim0, int dim1);

    Tensor softmax(int dim);

    Tensor logSoftmax(int dim);

    /**
     * Computes the arithmetic mean along one axis.
     *
     * <p>Negative axes count from the final dimension ({@code -1} is the last
     * axis). When {@code keepDim} is true, the reduced axis remains with size
     * one; otherwise it is removed. Implementations must reject axes outside
     * {@code [-rank, rank)} and empty reduction dimensions.</p>
     */
    Tensor mean(int dim, boolean keepDim);

    Tensor sum();

    /**
     * Computes the sum along one axis.
     *
     * <p>Negative axes count from the final dimension ({@code -1} is the last
     * axis). When {@code keepDim} is true, the reduced axis remains with size
     * one; otherwise it is removed. Implementations must reject axes outside
     * {@code [-rank, rank)}. Summing an empty reduction dimension yields zeros.</p>
     */
    Tensor sum(int dim, boolean keepDim);

    /**
     * Splits this tensor into {@code parts} equal, non-empty chunks along {@code axis}.
     *
     * <p>Negative axes count from the final dimension ({@code -1} is the last
     * axis). The axis size must be positive and evenly divisible by the positive
     * part count; requesting more parts than elements is invalid. The returned
     * tensors preserve chunk order.</p>
     */
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

    /** Exports logical I32 values from host-accessible storage in row-major order. */
    default int[] toIntArray() {
        throw new UnsupportedOperationException("Integer-array export is not supported by " + getClass().getSimpleName());
    }
}
