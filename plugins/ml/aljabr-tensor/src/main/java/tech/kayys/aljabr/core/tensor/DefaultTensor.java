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
        return record(OpType.MATMUL, new Tensor[] { this, other }, new Object[0], backend.matmul(this, other));
    }

    @Override
    public Tensor reshape(long... newShape) {
        return record(OpType.RESHAPE, new Tensor[] { this }, new Object[] { newShape.clone() }, backend.reshape(this, newShape));
    }

    @Override
    public Tensor softmax() {
        return record(OpType.SOFTMAX, new Tensor[] { this }, new Object[0], backend.softmax(this));
    }

    @Override
    public Tensor slice(long[] offsets, long[] sizes) {
        return record(OpType.SLICE, new Tensor[] { this }, new Object[] { offsets.clone(), sizes.clone() }, backend.slice(this, offsets, sizes));
    }

    @Override
    public Tensor pow(float exponent) {
        return record(OpType.POW, new Tensor[] { this }, new Object[] { exponent }, backend.pow(this, exponent));
    }

    @Override
    public Tensor mean() {
        return record(OpType.MEAN, new Tensor[] { this }, new Object[0], backend.mean(this));
    }

    @Override
    public Tensor abs() {
        return record(OpType.ABS, new Tensor[] { this }, new Object[0], backend.abs(this));
    }

    @Override
    public Tensor crossEntropy(Tensor t) {
        return record(OpType.CROSS_ENTROPY, new Tensor[] { this, t }, new Object[0], backend.crossEntropy(this, t));
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor t) {
        return record(OpType.BINARY_CROSS_ENTROPY, new Tensor[] { this, t }, new Object[0], backend.binaryCrossEntropy(this, t));
    }

    @Override
    public Tensor cast(DType dtype) {
        return record(OpType.CAST, new Tensor[] { this }, new Object[] { dtype }, backend.cast(this, dtype));
    }

    @Override
    public Tensor to(DeviceType device) {
        return record(OpType.TO, new Tensor[] { this }, new Object[] { device }, backend.to(this, device));
    }

    @Override
    public Tensor zerosLike() {
        return record(OpType.ZEROS_LIKE, new Tensor[] { this }, new Object[0], backend.zerosLike(this));
    }

    @Override
    public Tensor sqrt() {
        return record(OpType.SQRT, new Tensor[] { this }, new Object[0], backend.sqrt(this));
    }

    @Override
    public Tensor relu() {
        return record(OpType.RELU, new Tensor[] { this }, new Object[0], backend.relu(this));
    }

    @Override
    public Tensor gelu() {
        return record(OpType.GELU, new Tensor[] { this }, new Object[0], backend.gelu(this));
    }

    @Override
    public Tensor sigmoid() {
        return record(OpType.SIGMOID, new Tensor[] { this }, new Object[0], backend.sigmoid(this));
    }

    @Override
    public Tensor tanh() {
        return record(OpType.TANH, new Tensor[] { this }, new Object[0], backend.tanh(this));
    }

    @Override
    public Tensor log() {
        return record(OpType.LOG, new Tensor[] { this }, new Object[0], backend.log(this));
    }

    @Override
    public Tensor exp() {
        return record(OpType.EXP, new Tensor[] { this }, new Object[0], backend.exp(this));
    }

    @Override
    public Tensor silu() {
        return record(OpType.SILU, new Tensor[] { this }, new Object[0], backend.silu(this));
    }

    @Override
    public Tensor flatten() {
        return record(OpType.FLATTEN, new Tensor[] { this }, new Object[0], backend.flatten(this));
    }

    @Override
    public Tensor unsqueeze(int dim) {
        return record(OpType.UNSQUEEZE, new Tensor[] { this }, new Object[] { dim }, backend.unsqueeze(this, dim));
    }

    @Override
    public Tensor squeeze() {
        return record(OpType.SQUEEZE, new Tensor[] { this }, new Object[0], backend.squeeze(this));
    }

    @Override
    public Tensor transpose() {
        return record(OpType.TRANSPOSE, new Tensor[] { this }, new Object[0], backend.transpose(this));
    }

    @Override
    public Tensor transpose(int d0, int d1) {
        return record(OpType.TRANSPOSE, new Tensor[] { this }, new Object[] { d0, d1 }, backend.transpose(this, d0, d1));
    }

    @Override
    public Tensor softmax(int dim) {
        return record(OpType.SOFTMAX, new Tensor[] { this }, new Object[] { dim }, backend.softmax(this, dim));
    }

    @Override
    public Tensor logSoftmax(int dim) {
        return record(OpType.LOG_SOFTMAX, new Tensor[] { this }, new Object[] { dim }, backend.logSoftmax(this, dim));
    }

    @Override
    public Tensor mean(int dim, boolean keepDim) {
        return record(OpType.MEAN, new Tensor[] { this }, new Object[] { dim, keepDim }, backend.mean(this, dim, keepDim));
    }

    @Override
    public Tensor sum() {
        return record(OpType.SUM, new Tensor[] { this }, new Object[0], backend.sum(this));
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        return record(OpType.SUM, new Tensor[] { this }, new Object[] { dim, keepDim }, backend.sum(this, dim, keepDim));
    }

    @Override
    public java.util.List<Tensor> split(int axis, int parts) {
        ensureOpen();
        int rank = shape.rank();
        int normalizedAxis = axis < 0 ? rank + axis : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank) {
            throw new IllegalArgumentException(
                    "Split axis " + axis + " is out of range for tensor rank " + rank);
        }
        if (parts <= 0 || shape.dim(normalizedAxis) % parts != 0) {
            throw new IllegalArgumentException(
                    "Split dimension " + shape.dim(normalizedAxis)
                            + " must be evenly divisible by a positive part count; parts=" + parts);
        }

        java.util.List<Tensor> outputs = backend.split(this, normalizedAxis, parts);
        if (!AutogradContext.isRecordingEnabled()
                || !java.util.Arrays.stream(new Tensor[] { this }).anyMatch(Tensor::requiresGrad)) {
            return outputs;
        }

        if (outputs.size() != parts) {
            outputs.forEach(Tensor::release);
            throw new IllegalStateException(
                    "Backend returned " + outputs.size() + " split outputs; expected " + parts);
        }
        if (outputs.stream().anyMatch(output -> !(output instanceof DefaultTensor))) {
            outputs.forEach(Tensor::release);
            throw new UnsupportedOperationException(
                    "Autograd recording requires eager DefaultTensor split outputs");
        }

        long chunkSize = shape.dim(normalizedAxis) / parts;
        try {
            for (int index = 0; index < outputs.size(); index++) {
                Tensor output = outputs.get(index);
                DefaultTensor eagerOutput = (DefaultTensor) output;
                long[] offsets = new long[rank];
                long[] sizes = shape.dims();
                offsets[normalizedAxis] = Math.multiplyExact((long) index, chunkSize);
                sizes[normalizedAxis] = chunkSize;
                GraphNode inputNode = autogradNode != null ? autogradNode : GraphNode.constant(this);
                GraphNode node = new GraphNode(
                        OpType.SLICE,
                        java.util.List.of(inputNode),
                        new Object[] { offsets, sizes },
                        eagerOutput.shape(),
                        eagerOutput.dtype(),
                        OperationBackwardFunctions.forOperation(OpType.SLICE));
                node.setMaterializedData(output);
                eagerOutput.autogradNode = node;
            }
            return outputs;
        } catch (RuntimeException | Error failure) {
            outputs.forEach(Tensor::release);
            throw failure;
        }
    }

    @Override
    public Tensor layerNorm(long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        return record(OpType.LAYER_NORM, new Tensor[] { this, weight, bias }, new Object[] { normalizedShape.clone(), eps }, backend.layerNorm(this, normalizedShape, weight, bias, eps));
    }

    @Override
    public Tensor rmsNorm(Tensor weight, float eps) {
        return record(OpType.RMS_NORM, new Tensor[] { this, weight }, new Object[] { eps }, backend.rmsNorm(this, weight, eps));
    }

    @Override
    public Tensor batchNorm(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training,
            float momentum, float eps) {
        return record(OpType.BATCH_NORM, new Tensor[] { this, weight, bias, runningMean, runningVar }, new Object[] { training, momentum, eps }, backend.batchNorm(this, weight, bias, runningMean, runningVar, training, momentum, eps));
    }

    @Override
    public Tensor conv2d(Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) {
        return record(OpType.CONV2D, new Tensor[] { this, weight, bias }, new Object[] { stride, padding, dilation, groups }, backend.conv2d(this, weight, bias, stride, padding, dilation, groups));
    }

    @Override
    public Tensor maxPool2d(int kernelSize, int stride, int padding) {
        return record(OpType.MAX_POOL_2D, new Tensor[] { this }, new Object[] { kernelSize, stride, padding }, backend.maxPool2d(this, kernelSize, stride, padding));
    }

    @Override
    public Tensor adaptiveAvgPool2d(int outputH, int outputW) {
        return record(OpType.ADAPTIVE_AVG_POOL_2D, new Tensor[] { this }, new Object[] { outputH, outputW }, backend.adaptiveAvgPool2d(this, outputH, outputW));
    }

    @Override
    public Tensor dropout(float p, boolean training) {
        return record(OpType.DROPOUT, new Tensor[] { this }, new Object[] { p, training }, backend.dropout(this, p, training));
    }

    @Override
    public Tensor attention(Tensor K, Tensor V) {
        return record(OpType.ATTENTION, new Tensor[] { this, K, V }, new Object[0], backend.attention(this, K, V));
    }

    @Override
    public Tensor embedding(Tensor weight, long paddingIdx) {
        return record(OpType.EMBEDDING, new Tensor[] { this, weight }, new Object[] { paddingIdx }, backend.embedding(weight, this, paddingIdx));
    }

    @Override
    public Tensor upsampleNearest2d(int outputH, int outputW) {
        return record(OpType.UPSAMPLE_NEAREST_2D, new Tensor[] { this },
                new Object[] { outputH, outputW }, backend.upsampleNearest2d(this, outputH, outputW));
    }

    @Override
    public Tensor applyRoPE(int posOffset, float freqBase, boolean isNeox) {
        return record(OpType.ROPE, new Tensor[] { this }, new Object[] { posOffset, freqBase, isNeox }, backend.applyRoPE(this, posOffset, freqBase, isNeox));
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
                || java.util.Arrays.stream(inputs).filter(Objects::nonNull).noneMatch(Tensor::requiresGrad)) {
            return result;
        }
        if (!(result instanceof DefaultTensor eagerResult)) {
            result.release();
            throw new UnsupportedOperationException(
                    "Autograd recording requires an eager DefaultTensor result; got "
                            + result.getClass().getName());
        }
        java.util.List<GraphNode> inputNodes = java.util.Arrays.stream(inputs)
                .filter(Objects::nonNull)
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
            Tensor ownedGradient = grad;
            grad = null;
            if (ownedGradient != null && ownedGradient != this) {
                ownedGradient.release();
            }
            storage.release();
        }
    }

    private void ensureOpen() {
        if (released.get()) {
            throw new IllegalStateException("Tensor has already been released");
        }
    }
}
