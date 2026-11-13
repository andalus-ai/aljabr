package tech.kayys.aljabr.core.tensor;

import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.Buffer;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;
import tech.kayys.aljabr.core.tensor.lazy.GraphExecutor;
import tech.kayys.aljabr.core.tensor.lazy.GraphNode;
import tech.kayys.aljabr.core.tensor.lazy.OpType;
import tech.kayys.aljabr.core.tensor.lazy.OperationBackwardFunctions;

/**
 * Default eager tensor backed by backend-neutral {@link TensorStorage}.
 *
 * <p>The constructor takes ownership of one existing storage reference after
 * successful validation. The caller retains ownership if construction fails.
 * The legacy {@link #buffer()} accessor remains available for FFM/Arena-backed
 * buffers during the storage migration.</p>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class DefaultTensor implements Tensor {

    private final Shape shape;
    private final DType dtype;
    private final DeviceType device;
    private final TensorStorage storage;
    private final ComputeBackend backend;
    private Tensor grad;
    private boolean requiresGrad;
    private GraphNode autogradNode;
    private final AtomicBoolean released = new AtomicBoolean();

    /**
     * Creates a tensor using the legacy Buffer contract.
     *
     * <p>This overload is retained for source and binary compatibility.</p>
     */
    public DefaultTensor(Shape shape, DType dtype, DeviceType device, Buffer buffer, ComputeBackend backend) {
        this(shape, dtype, device, (TensorStorage) Objects.requireNonNull(buffer, "buffer must not be null"), backend);
    }

    /**
     * Creates a tensor using backend-neutral storage.
     *
     * <p>On successful construction this tensor owns the supplied storage
     * reference and releases it from {@link #release()}.</p>
     */
    public DefaultTensor(Shape shape, DType dtype, DeviceType device, TensorStorage storage, ComputeBackend backend) {
        this.shape = Objects.requireNonNull(shape, "shape must not be null");
        this.dtype = Objects.requireNonNull(dtype, "dtype must not be null");
        this.device = Objects.requireNonNull(device, "device must not be null");
        this.storage = Objects.requireNonNull(storage, "storage must not be null");
        this.backend = Objects.requireNonNull(backend, "backend must not be null");

        // Larger allocations are valid for pooled storage and tensor views.
        // A tensor must never expose less capacity than its logical shape needs.
        final long requiredBytes = dtype.memoryFootprintBytes(shape.numel());
        final long availableBytes = storage.sizeBytes();
        if (availableBytes < requiredBytes) {
            throw new IllegalArgumentException(
                    "Tensor storage is too small: shape=" + shape
                            + ", dtype=" + dtype
                            + ", requires=" + requiredBytes + " bytes"
                            + ", available=" + availableBytes + " bytes");
        }
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

    /** Returns the canonical backend-neutral storage contract. */
    @Override
    public TensorStorage storage() {
        ensureOpen();
        return storage;
    }

    /**
     * Returns the legacy FFM/Arena buffer view.
     *
     * @throws UnsupportedOperationException when this tensor is backed by
     *         storage that does not implement the legacy Buffer contract
     */
    public Buffer buffer() {
        ensureOpen();
        if (storage instanceof Buffer buffer) {
            return buffer;
        }
        throw new UnsupportedOperationException(
                "Tensor storage does not implement the legacy Buffer API: "
                        + storage.getClass().getName()
                        + ". Use storage() and check for HostAccessibleStorage when host access is required.");
    }

    private MemorySegment requireHostSegment(String operation) {
        ensureOpen();
        if (storage instanceof HostAccessibleStorage hostStorage) {
            return Objects.requireNonNull(hostStorage.segment(),
                    "HostAccessibleStorage.segment() must not return null");
        }
        throw new UnsupportedOperationException(
                operation + " requires host-accessible tensor storage; device=" + device
                        + ", storage=" + storage.getClass().getName()
                        + ". Transfer the tensor to host-accessible storage before reading its values.");
    }

    @Override
    public Tensor add(Tensor other) {
        return record(OpType.ADD, new Tensor[] { this, other }, new Object[0], backend.add(this, other));
    }

    @Override
    public Tensor sub(Tensor other) {
        return record(OpType.SUB, new Tensor[] { this, other }, new Object[0], backend.sub(this, other));
    }

    @Override
    public Tensor mul(Tensor other) {
        return record(OpType.MUL, new Tensor[] { this, other }, new Object[0], backend.mul(this, other));
    }

    @Override
    public Tensor mul(float scalar) {
        return record(OpType.MUL_SCALAR, new Tensor[] { this }, new Object[] { scalar }, backend.mul(this, scalar));
    }

    @Override
    public Tensor div(float scalar) {
        return record(OpType.DIV_SCALAR, new Tensor[] { this }, new Object[] { scalar }, backend.div(this, scalar));
    }

    @Override
    public Tensor div(Tensor other) {
        return record(OpType.DIV, new Tensor[] { this, other }, new Object[0], backend.div(this, other));
    }

    @Override
    public Tensor add(float scalar) {
        return record(OpType.ADD_SCALAR, new Tensor[] { this }, new Object[] { scalar }, backend.addScalar(this, scalar));
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
        if (dtype != DType.F32) {
            throw new UnsupportedOperationException(
                    "toFloatArray currently requires F32 tensors; actual dtype is " + dtype);
        }
        return requireHostSegment("toFloatArray")
                .asSlice(0, Math.multiplyExact(numel(), Float.BYTES))
                .toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
    }

    @Override
    public float item() {
        if (numel() != 1) {
            throw new IllegalStateException("item() requires exactly one element; tensor has " + numel());
        }
        return switch (dtype) {
            case F32 -> requireHostSegment("item()").get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
            case I32 -> requireHostSegment("item()").get(java.lang.foreign.ValueLayout.JAVA_INT, 0);
            case I8, INT8 -> requireHostSegment("item()").get(java.lang.foreign.ValueLayout.JAVA_BYTE, 0);
            default -> throw new UnsupportedOperationException("item() is not supported for dtype " + dtype);
        };
    }

    @Override
    public void backward() {
        ensureOpen();
        if (!requiresGrad()) {
            return;
        }
        if (numel() != 1) {
            throw new UnsupportedOperationException(
                    "Implicit backward() requires a scalar output; use backward(Tensor outputGradient) "
                            + "for non-scalar tensors");
        }
        Tensor zero = zerosLike();
        Tensor seed = null;
        try {
            seed = zero.add(1.0f);
            backward(seed);
        } finally {
            zero.release();
            if (seed != null) {
                seed.release();
            }
        }
    }

    @Override
    public void backward(Tensor outputGradient) {
        ensureOpen();
        Objects.requireNonNull(outputGradient, "outputGradient");
        if (!requiresGrad()) {
            return;
        }
        if (!shape.equals(outputGradient.shape())) {
            throw new IllegalArgumentException("Output gradient shape " + outputGradient.shape()
                    + " does not match tensor shape " + shape);
        }
        GraphNode root = autogradNode != null ? autogradNode : GraphNode.constant(this);
        root.setMaterializedData(this);
        try (AutogradContext.Scope ignored = AutogradContext.pauseRecording()) {
            new GraphExecutor(backend).backward(root, outputGradient);
        }
    }

    private Tensor record(OpType operation, Tensor[] inputs, Object[] arguments, Tensor result) {
        if (!AutogradContext.isRecordingEnabled()
                || result == null
                || java.util.Arrays.stream(inputs).noneMatch(Tensor::requiresGrad)) {
            return result;
        }
        if (!(result instanceof DefaultTensor eagerResult)) {
            result.release();
            throw new UnsupportedOperationException(
                    "Autograd recording requires an eager DefaultTensor result; got "
                            + result.getClass().getName());
        }
        java.util.List<GraphNode> inputNodes = java.util.Arrays.stream(inputs)
                .map(input -> input instanceof DefaultTensor eager && eager.autogradNode != null
                        ? eager.autogradNode
                        : GraphNode.constant(input))
                .toList();
        GraphNode node = new GraphNode(operation, inputNodes, arguments, result.shape(), result.dtype(),
                OperationBackwardFunctions.forOperation(operation));
        node.setMaterializedData(result);
        eagerResult.autogradNode = node;
        return result;
    }

    /** Internal graph association used when an eager result feeds another operation. */
    public GraphNode autogradNode() {
        return autogradNode;
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
        return requiresGrad || (autogradNode != null && autogradNode.requiresGrad());
    }

    @Override
    public void setRequiresGrad(boolean r) {
        if (autogradNode != null && autogradNode.getOp() != OpType.CONSTANT) {
            throw new IllegalStateException("requiresGrad can only be set on a leaf tensor");
        }
        this.requiresGrad = r;
    }

    @Override
    public void release() {
        if (released.compareAndSet(false, true)) {
            storage.release();
        }
    }

    private void ensureOpen() {
        if (released.get()) {
            throw new IllegalStateException("Tensor has already been released");
        }
    }
}
