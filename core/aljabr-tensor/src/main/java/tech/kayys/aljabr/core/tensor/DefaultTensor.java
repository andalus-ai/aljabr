package tech.kayys.aljabr.core.tensor;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.Buffer;

/**
 * Default concrete tensor backed by a {@link Buffer}.
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class DefaultTensor implements Tensor {

    private final Shape shape;
    private final DType dtype;
    private final DeviceType device;
    private final Buffer buffer;
    private final ComputeBackend backend;
    private Tensor grad;
    private boolean requiresGrad;

    public DefaultTensor(Shape shape, DType dtype, DeviceType device, Buffer buffer, ComputeBackend backend) {
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
        this.buffer = buffer;
        this.backend = backend;
    }

    @Override
    public Tensor eval() {
        return this;
    }

    @Override
    public Shape shape() {
        return shape;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public DeviceType device() {
        return device;
    }

    @Override
    public ComputeBackend backend() {
        return backend;
    }

    public Buffer buffer() {
        return buffer;
    }

    @Override
    public Tensor add(Tensor other) {
        return backend.add(this, other);
    }

    @Override
    public Tensor sub(Tensor other) {
        return backend.sub(this, other);
    }

    @Override
    public Tensor mul(Tensor other) {
        return backend.mul(this, other);
    }

    @Override
    public Tensor mul(float scalar) {
        return backend.mul(this, scalar);
    }

    @Override
    public Tensor div(float scalar) {
        return backend.div(this, scalar);
    }

    @Override
    public Tensor div(Tensor other) {
        return backend.div(this, other);
    }

    @Override
    public Tensor add(float scalar) {
        return backend.addScalar(this, scalar);
    }

    @Override
    public Tensor matmul(Tensor other) {
        return backend.matmul(this, other);
    }

    @Override
    public Tensor reshape(long... newShape) {
        return backend.reshape(this, newShape);
    }

    @Override
    public Tensor softmax() {
        return backend.softmax(this);
    }

    @Override
    public Tensor slice(long[] offsets, long[] sizes) {
        return backend.slice(this, offsets, sizes);
    }

    @Override
    public Tensor pow(float exponent) {
        return backend.pow(this, exponent);
    }

    @Override
    public Tensor mean() {
        return backend.mean(this);
    }

    @Override
    public Tensor abs() {
        return backend.abs(this);
    }

    @Override
    public Tensor crossEntropy(Tensor t) {
        return backend.crossEntropy(this, t);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor t) {
        return backend.binaryCrossEntropy(this, t);
    }

    @Override
    public Tensor cast(DType dtype) {
        return backend.cast(this, dtype);
    }

    @Override
    public Tensor to(DeviceType device) {
        return backend.to(this, device);
    }

    @Override
    public Tensor zerosLike() {
        return backend.zerosLike(this);
    }

    @Override
    public Tensor sqrt() {
        return backend.sqrt(this);
    }

    @Override
    public Tensor relu() {
        return backend.relu(this);
    }

    @Override
    public Tensor gelu() {
        return backend.gelu(this);
    }

    @Override
    public Tensor sigmoid() {
        return backend.sigmoid(this);
    }

    @Override
    public Tensor tanh() {
        return backend.tanh(this);
    }

    @Override
    public Tensor log() {
        return backend.log(this);
    }

    @Override
    public Tensor exp() {
        return backend.exp(this);
    }

    @Override
    public Tensor silu() {
        return backend.silu(this);
    }

    @Override
    public Tensor flatten() {
        return backend.flatten(this);
    }

    @Override
    public Tensor unsqueeze(int dim) {
        return backend.unsqueeze(this, dim);
    }

    @Override
    public Tensor squeeze() {
        return backend.squeeze(this);
    }

    @Override
    public Tensor transpose() {
        return backend.transpose(this);
    }

    @Override
    public Tensor transpose(int d0, int d1) {
        return backend.transpose(this, d0, d1);
    }

    @Override
    public Tensor softmax(int dim) {
        return backend.softmax(this, dim);
    }

    @Override
    public Tensor logSoftmax(int dim) {
        return backend.logSoftmax(this, dim);
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        return backend.mean(this, dim, keepDim);
    }

    @Override
    public Tensor sum() {
        return backend.sum(this);
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        return backend.sum(this, dim, keepDim);
    }

    @Override
    public java.util.List<Tensor> split(int axis, int parts) {
        return backend.split(this, axis, parts);
    }

    @Override
    public Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        return backend.layerNorm(this, normalizedShape, weight, bias, eps);
    }

    @Override
    public Tensor rmsNorm(Tensor weight, float eps) {
        return backend.rmsNorm(this, weight, eps);
    }

    @Override
    public Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training,
            float momentum, float eps) {
        return backend.batchNorm(this, weight, bias, runningMean, runningVar, training, momentum, eps);
    }

    @Override
    public Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return backend.conv2d(this, weight, bias, stride, padding, dilation, groups);
    }

    @Override
    public Tensor maxPool2d(int kernelSize, int stride, int padding) {
        return backend.maxPool2d(this, kernelSize, stride, padding);
    }

    @Override
    public Tensor adaptiveAvgPool2d(int outputH, int outputW) {
        return backend.adaptiveAvgPool2d(this, outputH, outputW);
    }

    @Override
    public Tensor dropout(float p, boolean training) {
        return backend.dropout(this, p, training);
    }

    @Override
    public Tensor attention(Tensor K, Tensor V) {
        return backend.attention(this, K, V);
    }

    @Override
    public Tensor embedding(Tensor weight, long paddingIdx) {
        return backend.embedding(weight, this, paddingIdx);
    }

    @Override
    public Tensor applyRoPE(int posOffset, float freqBase, boolean isNeox) {
        return backend.applyRoPE(this, posOffset, freqBase, isNeox);
    }

    @Override
    public long numel() {
        return shape.numel();
    }

    @Override
    public float[] toFloatArray() {
        return buffer.segment().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
    }

    @Override
    public float item() {
        return 0f;
    }

    @Override
    public void backward() {
    }

    @Override
    public Tensor grad() {
        return grad;
    }

    @Override
    public void setGrad(Tensor g) {
        this.grad = g;
    }

    @Override
    public boolean requiresGrad() {
        return requiresGrad;
    }

    @Override
    public void setRequiresGrad(boolean r) {
        this.requiresGrad = r;
    }

    @Override
    public void release() {
        if (buffer != null)
            buffer.release();
    }
}
