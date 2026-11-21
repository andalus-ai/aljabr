package tech.kayys.aljabr.core.tensor;

import java.lang.foreign.MemorySegment;
import java.util.Map;
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
import tech.kayys.aljabr.diagnostics.Diagnostic;
import tech.kayys.aljabr.diagnostics.DiagnosticSeverity;
import tech.kayys.aljabr.diagnostics.DiagnosticSink;
import tech.kayys.aljabr.core.tensor.internal.TensorOperationValidator;

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
    private final TensorLayout layout;
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
     * <p>On successful construction this tensor takes ownership of the one
     * existing storage reference supplied by the caller and releases that
     * reference from {@link #release()}. The caller must not release that
     * transferred reference afterward. If validation or construction fails,
     * ownership remains with the caller. Call {@link TensorStorage#retain()}
     * first when the caller also needs to keep an independent reference.</p>
     */
    public DefaultTensor(Shape shape, DType dtype, DeviceType device, TensorStorage storage, ComputeBackend backend) {
        this(shape, dtype, device, storage, backend, DiagnosticSink.noop());
    }

    /**
     * Creates a tensor and reports construction-time validation failures to an
     * explicitly supplied diagnostic sink.
     *
     * <p>The sink is consulted only when validation fails. Successful tensor
     * construction does not create diagnostics or invoke the sink. Existing
     * constructors remain silent and source-compatible.</p>
     *
     * @param diagnosticSink receives structured validation failures
     */
    public DefaultTensor(
            Shape shape,
            DType dtype,
            DeviceType device,
            TensorStorage storage,
            ComputeBackend backend,
            DiagnosticSink diagnosticSink) {
        this(shape, dtype, device, storage, backend, TensorLayout.contiguous(shape), diagnosticSink);
    }

    /**
     * Creates a tensor with an explicit immutable layout.
     *
     * <p>The layout is metadata only: operations must still be implemented by
     * a backend that understands the supplied strides. Existing constructors
     * continue to create canonical contiguous tensors.</p>
     *
     * <p>On successful construction this tensor takes ownership of the one
     * existing storage reference. If validation fails, ownership remains with
     * the caller.</p>
     */
    public DefaultTensor(
            Shape shape,
            DType dtype,
            DeviceType device,
            TensorStorage storage,
            ComputeBackend backend,
            TensorLayout layout) {
        this(shape, dtype, device, storage, backend, layout, DiagnosticSink.noop());
    }

    /**
     * Creates a tensor with an explicit layout and diagnostic sink.
     *
     * <p>Block-quantized tensors currently require canonical contiguous
     * layouts with zero storage offset because their storage addressing is
     * block-based rather than element-strided.</p>
     */
    public DefaultTensor(
            Shape shape,
            DType dtype,
            DeviceType device,
            TensorStorage storage,
            ComputeBackend backend,
            TensorLayout layout,
            DiagnosticSink diagnosticSink) {
        this.shape = Objects.requireNonNull(shape, "shape must not be null");
        this.dtype = Objects.requireNonNull(dtype, "dtype must not be null");
        this.device = Objects.requireNonNull(device, "device must not be null");
        this.storage = Objects.requireNonNull(storage, "storage must not be null");
        this.backend = Objects.requireNonNull(backend, "backend must not be null");
        this.layout = Objects.requireNonNull(layout, "layout must not be null");
        Objects.requireNonNull(diagnosticSink, "diagnosticSink must not be null");

        if (!shape.equals(layout.shape())) {
            throw new IllegalArgumentException(
                    "Tensor layout shape does not match tensor shape: shape=" + shape
                            + ", layoutShape=" + layout.shape());
        }

        final long requiredBytes;
        try {
            if (dtype.isQuantized()) {
                if (!layout.isContiguous() || layout.storageOffset() != 0) {
                    throw new IllegalArgumentException(
                            "Block-quantized tensors currently require a contiguous layout with zero storage offset");
                }
                requiredBytes = dtype.memoryFootprintBytes(shape.numel());
            } else {
                requiredBytes = dtype.memoryFootprintBytes(layout.requiredStorageElements());
            }
        } catch (IllegalArgumentException failure) {
            diagnosticSink.accept(new Diagnostic(
                    "tensor.dtype.element-count-invalid",
                    DiagnosticSeverity.ERROR,
                    "tensor.validation",
                    failure.getMessage(),
                    Map.of(
                            "shape", shape.toString(),
                            "layout", layout.toString(),
                            "dtype", dtype.name(),
                            "numel", Long.toString(shape.numel()),
                            "causeType", failure.getClass().getName())));
            throw failure;
        }

        final long availableBytes = storage.sizeBytes();
        if (availableBytes < requiredBytes) {
            String message = "Tensor storage is too small: shape=" + shape
                    + ", layout=" + layout
                    + ", dtype=" + dtype
                    + ", requires=" + requiredBytes + " bytes"
                    + ", available=" + availableBytes + " bytes";
            diagnosticSink.accept(new Diagnostic(
                    "tensor.storage.capacity-insufficient",
                    DiagnosticSeverity.ERROR,
                    "tensor.validation",
                    message,
                    Map.of(
                            "shape", shape.toString(),
                            "layout", layout.toString(),
                            "dtype", dtype.name(),
                            "device", device.name(),
                            "requiredBytes", Long.toString(requiredBytes),
                            "availableBytes", Long.toString(availableBytes))));
            throw new IllegalArgumentException(message);
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

    /** Returns this tensor's immutable element-based storage layout. */
    @Override
    public TensorLayout layout() {
        return layout;
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

    /**
     * Returns the canonical backend-neutral storage contract as a borrowed
     * reference. The tensor continues to own the storage; callers must not
     * release this reference. Retain it explicitly if it must outlive this
     * tensor.
     */
    @Override
    public TensorStorage storage() {
        ensureOpen();
        return storage;
    }

    /**
     * Returns the legacy FFM/Arena buffer as a borrowed view. The returned
     * buffer does not carry an independent ownership reference and must not be
     * released by the caller.
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
        TensorOperationValidator.validateReshape(shape, newShape);
        return record(OpType.RESHAPE, new Tensor[] { this }, new Object[] { newShape.clone() }, backend.reshape(this, newShape));
    }

    @Override
    public Tensor softmax() {
        return record(OpType.SOFTMAX, new Tensor[] { this }, new Object[0], backend.softmax(this));
    }

    @Override
    public Tensor slice(long[] offsets, long[] sizes) {
        TensorOperationValidator.validateSlice(shape, offsets, sizes);
        return record(OpType.SLICE, new Tensor[] { this }, new Object[] { offsets.clone(), sizes.clone() }, backend.slice(this, offsets, sizes));
    }

    @Override
    public Tensor pow(float exponent) {
        return record(OpType.POW, new Tensor[] { this }, new Object[] { exponent }, backend.pow(this, exponent));
    }

    @Override
    public Tensor mean() {
        ensureOpen();
        if (shape.numel() == 0) {
            throw new IllegalArgumentException("Mean is undefined for an empty tensor");
        }
        return record(OpType.MEAN, new Tensor[] { this }, new Object[0], backend.mean(this));
    }

    @Override
    public Tensor sign() {
        return record(OpType.SIGN, new Tensor[] { this }, new Object[0], backend.sign(this));
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
        ensureOpen();
        TensorOperationValidator.validateFlatten(shape);
        return record(OpType.FLATTEN, new Tensor[] { this }, new Object[0], backend.flatten(this));
    }

    @Override
    public Tensor unsqueeze(int dim) {
        ensureOpen();
        TensorOperationValidator.UnsqueezePlan plan =
                TensorOperationValidator.validateUnsqueeze(shape, dim);
        return record(OpType.UNSQUEEZE, new Tensor[] { this }, new Object[] { dim },
                backend.unsqueeze(this, plan.axis()));
    }

    @Override
    public Tensor squeeze() {
        ensureOpen();
        return record(OpType.SQUEEZE, new Tensor[] { this }, new Object[0], backend.squeeze(this));
    }

    @Override
    public Tensor transpose() {
        ensureOpen();
        TensorOperationValidator.validateTranspose(shape);
        return record(OpType.TRANSPOSE, new Tensor[] { this }, new Object[0], backend.transpose(this));
    }

    @Override
    public Tensor transpose(int d0, int d1) {
        ensureOpen();
        TensorOperationValidator.TransposePlan plan =
                TensorOperationValidator.validateTranspose(shape, d0, d1);
        return record(OpType.TRANSPOSE, new Tensor[] { this },
                new Object[] { plan.axis0(), plan.axis1() },
                backend.transpose(this, plan.axis0(), plan.axis1()));
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
        ensureOpen();
        int axis = TensorOperationValidator.validateReduction(shape, dim, true).axis();
        return record(OpType.MEAN, new Tensor[] { this }, new Object[] { axis, keepDim }, backend.mean(this, axis, keepDim));
    }

    @Override
    public Tensor sum() {
        return record(OpType.SUM, new Tensor[] { this }, new Object[0], backend.sum(this));
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        ensureOpen();
        int axis = TensorOperationValidator.validateReduction(shape, dim, false).axis();
        return record(OpType.SUM, new Tensor[] { this }, new Object[] { axis, keepDim }, backend.sum(this, axis, keepDim));
    }

    /**
     * Concatenates eager tensors and records a multi-input graph node when any
     * source participates in autograd.
     */
    public static Tensor cat(java.util.List<Tensor> tensors, int axis) {
        return cat(tensors, axis, DiagnosticSink.noop());
    }

    /** Concatenates eager tensors and reports input contract failures to the supplied sink. */
    public static Tensor cat(java.util.List<Tensor> tensors, int axis, DiagnosticSink diagnosticSink) {
        Objects.requireNonNull(tensors, "tensors must not be null");
        for (int i = 0; i < tensors.size(); i++) {
            Tensor tensor = Objects.requireNonNull(tensors.get(i), "tensor at index " + i + " must not be null");
            if (!(tensor instanceof DefaultTensor eager)) {
                throw new UnsupportedOperationException(
                        "Concatenation currently requires eager DefaultTensor inputs");
            }
            eager.ensureOpen();
        }
        TensorOperationValidator.CatPlan plan =
                TensorOperationValidator.validateCat(tensors, axis, diagnosticSink);
        boolean requiresGrad = tensors.stream().anyMatch(Tensor::requiresGrad);
        Tensor result = plan.first().backend().cat(tensors, plan.axis());
        if (!AutogradContext.isRecordingEnabled() || !requiresGrad) return result;
        if (!(result instanceof DefaultTensor eagerResult)) {
            result.release();
            throw new UnsupportedOperationException(
                    "Autograd recording requires an eager DefaultTensor concatenation result");
        }
        java.util.List<GraphNode> inputNodes = tensors.stream()
                .map(tensor -> {
                    DefaultTensor eager = (DefaultTensor) tensor;
                    return eager.autogradNode != null ? eager.autogradNode : GraphNode.constant(tensor);
                }).toList();
        GraphNode node = new GraphNode(OpType.CAT, inputNodes, new Object[] { plan.axis() },
                eagerResult.shape(), eagerResult.dtype(), OperationBackwardFunctions.forOperation(OpType.CAT));
        node.setMaterializedData(result);
        eagerResult.autogradNode = node;
        return result;
    }

    @Override
    public java.util.List<Tensor> split(int axis, int parts) {
        ensureOpen();
        TensorOperationValidator.SplitPlan plan =
                TensorOperationValidator.validateSplit(shape, axis, parts);
        int normalizedAxis = plan.axis();

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

        long chunkSize = plan.chunkSize();
        try {
            for (int index = 0; index < outputs.size(); index++) {
                Tensor output = outputs.get(index);
                DefaultTensor eagerOutput = (DefaultTensor) output;
                long[] offsets = new long[shape.rank()];
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
        ensureOpen();
        if (dtype != DType.F32) {
            throw new UnsupportedOperationException(
                    "toFloatArray currently requires F32 tensors; actual dtype is " + dtype);
        }
        int length = Math.toIntExact(numel());
        float[] values = new float[length];
        if (length == 0) {
            return values;
        }

        MemorySegment segment = requireHostSegment("toFloatArray");
        long[] strides = layout.strides();
        if (layout.isContiguous()) {
            long byteOffset = Math.multiplyExact(layout.storageOffset(), Float.BYTES);
            return segment.asSlice(byteOffset, Math.multiplyExact((long) length, Float.BYTES))
                    .toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
        }

        for (int linearIndex = 0; linearIndex < length; linearIndex++) {
            long elementOffset = elementOffsetForLinearIndex(linearIndex, strides);
            long byteOffset = Math.multiplyExact(elementOffset, Float.BYTES);
            values[linearIndex] = segment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, byteOffset);
        }
        return values;
    }

    @Override
    public int[] toIntArray() {
        ensureOpen();
        if (dtype != DType.I32) {
            throw new UnsupportedOperationException(
                    "toIntArray currently requires I32 tensors; actual dtype is " + dtype);
        }
        int length = Math.toIntExact(numel());
        int[] values = new int[length];
        if (length == 0) {
            return values;
        }

        MemorySegment segment = requireHostSegment("toIntArray");
        long[] strides = layout.strides();
        if (layout.isContiguous()) {
            long byteOffset = Math.multiplyExact(layout.storageOffset(), Integer.BYTES);
            return segment.asSlice(byteOffset, Math.multiplyExact((long) length, Integer.BYTES))
                    .toArray(java.lang.foreign.ValueLayout.JAVA_INT);
        }

        for (int linearIndex = 0; linearIndex < length; linearIndex++) {
            long elementOffset = elementOffsetForLinearIndex(linearIndex, strides);
            long byteOffset = Math.multiplyExact(elementOffset, Integer.BYTES);
            values[linearIndex] = segment.get(java.lang.foreign.ValueLayout.JAVA_INT, byteOffset);
        }
        return values;
    }

    /**
     * Maps a row-major logical element index to its backing-storage element
     * offset. The caller must supply a live tensor and a valid logical index.
     */
    private long elementOffsetForLinearIndex(long linearIndex, long[] strides) {
        long offset = layout.storageOffset();
        for (int dimension = shape.rank() - 1; dimension >= 0; dimension--) {
            long size = shape.dim(dimension);
            long coordinate = linearIndex % size;
            linearIndex /= size;
            offset = Math.addExact(offset, Math.multiplyExact(coordinate, strides[dimension]));
        }
        return offset;
    }

    @Override
    public float item() {
        ensureOpen();
        if (numel() != 1) {
            throw new IllegalStateException("item() requires exactly one element; tensor has " + numel());
        }
        long byteOffset = Math.multiplyExact(layout.storageOffset(), dtype.blockByteSize());
        MemorySegment segment = requireHostSegment("item()");
        return switch (dtype) {
            case F32 -> segment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, byteOffset);
            case I32 -> segment.get(java.lang.foreign.ValueLayout.JAVA_INT, byteOffset);
            case I8, INT8 -> segment.get(java.lang.foreign.ValueLayout.JAVA_BYTE, byteOffset);
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

    /**
     * Returns the currently owned gradient as a borrowed reference.
     * Callers must not release it.
     */
    @Override
    public Tensor grad() {
        return grad;
    }

    /**
     * Transfers ownership of {@code g} to this tensor. Replacing the current
     * gradient releases the previous gradient; passing {@code null} clears it.
     */
    @Override
    public void setGrad(Tensor g) {
        ensureOpen();
        if (g == this) {
            throw new IllegalArgumentException("A tensor cannot own itself as its gradient");
        }
        Tensor previous = this.grad;
        if (previous == g) {
            return;
        }
        this.grad = g;
        if (previous != null) {
            previous.release();
        }
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

    /**
     * Releases this tensor's storage reference and its owned gradient exactly
     * once. Any borrowed storage, buffer, or gradient reference obtained from
     * this tensor must no longer be used after release.
     */
    @Override
    public void release() {
        if (released.compareAndSet(false, true)) {
            Tensor ownedGradient = grad;
            grad = null;

            RuntimeException runtimeFailure = null;
            Error errorFailure = null;
            if (ownedGradient != null && ownedGradient != this) {
                try {
                    ownedGradient.release();
                } catch (RuntimeException failure) {
                    runtimeFailure = failure;
                } catch (Error failure) {
                    errorFailure = failure;
                }
            }

            try {
                storage.release();
            } catch (RuntimeException failure) {
                if (runtimeFailure != null) {
                    runtimeFailure.addSuppressed(failure);
                } else if (errorFailure != null) {
                    errorFailure.addSuppressed(failure);
                } else {
                    runtimeFailure = failure;
                }
            } catch (Error failure) {
                if (runtimeFailure != null) {
                    runtimeFailure.addSuppressed(failure);
                } else if (errorFailure != null) {
                    errorFailure.addSuppressed(failure);
                } else {
                    errorFailure = failure;
                }
            }

            if (runtimeFailure != null) {
                throw runtimeFailure;
            }
            if (errorFailure != null) {
                throw errorFailure;
            }
        }
    }

    private void ensureOpen() {
        if (released.get()) {
            throw new IllegalStateException("Tensor has already been released");
        }
    }
}
