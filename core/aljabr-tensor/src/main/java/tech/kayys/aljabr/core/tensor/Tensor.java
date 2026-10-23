package tech.kayys.aljabr.core.tensor;

import tech.kayys.aljabr.core.backend.ComputeBackend;

/**
 * The core abstraction of the Aljabr framework representing a multi-dimensional
 * array of elements.
 * 
 * <p>
 * A {@code Tensor} is immutable with respect to its mathematical properties
 * (shape, data type).
 * By default, computation operations return a new {@code Tensor} instance
 * unless an explicitly
 * mutable operation (like {@code assign} or in-place ops if added) is used.
 * 
 * <p>
 * Tensors are backed by a {@link ComputeBackend} and allocate memory according
 * to the rules of
 * that backend. For example, a CPU backend allocates off-heap native memory
 * using the Java Foreign
 * Function &amp; Memory API (FFM), while a GPU backend allocates memory on the
 * device.
 * @author bhangun
 */
public interface Tensor {
    /**
     * Evaluates and materializes this Tensor if it is a lazy computation graph
     * node.
     * For eager tensors, this returns the tensor itself.
     */
    Tensor eval();

    Shape shape();

    DeviceType device();

    DType dtype();

    ComputeBackend backend();

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

    static Tensor zeros(long... shape) {
        return TensorFactory.zeros(shape);
    }

    static Tensor ones(long... shape) {
        return TensorFactory.ones(shape);
    }

    static Tensor full(float value, long... shape) {
        return TensorFactory.full(value, shape);
    }

    static Tensor of(float[] data, long... shape) {
        return TensorFactory.of(data, shape);
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
