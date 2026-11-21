package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.backend.cpu.ops.CpuOps;
import tech.kayys.aljabr.backend.cpu.ops.MatmulCpu;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.tensor.internal.TensorOperationValidator;
import tech.kayys.aljabr.diagnostics.DiagnosticSink;
import tech.kayys.aljabr.core.memory.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pure-Java CPU backend for the Aljabr compute engine.
 *
 * <p>All pointwise operations delegate to {@link CpuOps}, which uses the JDK
 * Incubator Vector API where available. An optional {@link OffHeapBufferPool}
 * recycles off-heap memory across execution steps.</p>
 */
public final class CpuBackend implements ComputeBackend {

    /** Defines whether this backend is responsible for closing an injected pool. */
    public enum PoolOwnership { BORROWED, OWNED }

    private final OffHeapBufferPool pool;
    private final PoolOwnership poolOwnership;
    private final TensorAllocator outputAllocator;
    private final AtomicBoolean closed = new AtomicBoolean();

    public CpuBackend() {
        this(null, PoolOwnership.BORROWED, null);
    }

    /** Injects a shared pool; the caller retains responsibility for closing it. */
    public CpuBackend(OffHeapBufferPool pool) {
        this(pool, PoolOwnership.BORROWED, null);
    }

    /** Injects a pool with an explicit lifecycle ownership policy. */
    public CpuBackend(OffHeapBufferPool pool, PoolOwnership poolOwnership) {
        this(pool, poolOwnership, null);
    }

    /**
     * Injects a CPU output-storage allocator independently of the pool used by
     * the legacy {@link #allocate(long)} API. The allocator must return
     * host-accessible storage because current CPU kernels operate on
     * {@link MemorySegment} views.
     */
    public CpuBackend(
            OffHeapBufferPool pool,
            PoolOwnership poolOwnership,
            TensorAllocator outputAllocator) {
        this.pool = pool;
        this.poolOwnership = Objects.requireNonNull(poolOwnership, "poolOwnership");
        if (pool == null && poolOwnership == PoolOwnership.OWNED) {
            throw new IllegalArgumentException("An owned pool must not be null");
        }
        this.outputAllocator = outputAllocator != null ? outputAllocator : this::allocate;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)
                && pool != null && poolOwnership == PoolOwnership.OWNED) {
            pool.close();
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * Allocates storage for CPU kernel outputs using the configured policy.
     * CPU kernels currently require host-accessible segments; device-local
     * storage belongs to a different backend implementation.
     */
    private HostAccessibleStorage allocateOutput(long byteSize) {
        if (closed.get()) {
            throw new IllegalStateException("CPU backend is closed");
        }
        TensorStorage storage = Objects.requireNonNull(
                outputAllocator.allocate(byteSize), "CPU output allocator must not return null");
        if (!(storage instanceof HostAccessibleStorage hostStorage)) {
            storage.release();
            throw new IllegalArgumentException(
                    "CPU output allocator must return HostAccessibleStorage, got "
                            + storage.getClass().getName());
        }
        if (hostStorage.sizeBytes() < byteSize) {
            long available = hostStorage.sizeBytes();
            hostStorage.release();
            throw new IllegalStateException(
                    "CPU output allocator returned " + available + " bytes; " + byteSize + " required");
        }
        return hostStorage;
    }

    @Override
    public TensorStorage allocateStorage(long byteSize) {
        return allocateOutput(byteSize);
    }

    @Override
    public CpuBuffer allocate(long byteSize) {
        if (closed.get()) {
            throw new IllegalStateException("CPU backend is closed");
        }
        if (pool != null) {
            return new CpuBuffer(pool.acquire(byteSize), pool.arena(), pool);
        }
        return new CpuBuffer(byteSize);
    }

    private static HostAccessibleStorage hostStorage(Tensor tensor) {
        return CpuTensorStorageAccess.hostStorage(tensor);
    }

    private static MemorySegment seg(Tensor tensor) {
        return CpuTensorStorageAccess.segment(tensor);
    }

    private DefaultTensor out(Shape shape, HostAccessibleStorage buf) {
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, buf, this);
    }
    // ── Arithmetic ────────────────────────────────────────────────────────────

    @Override
    public Tensor add(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 0);
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.binary(a, b, o.segment(), n, CpuOps::add, (x, y) -> x + y);
        return out(s, o);
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 3);
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.binary(a, b, o.segment(), n, CpuOps::sub, (x, y) -> x - y);
        return out(s, o);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, (src, dst, count) -> CpuOps.mulScalar(src, scalar, dst, count), x -> x * scalar);
        return out(s, o);
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 1);
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.binary(a, b, o.segment(), n, CpuOps::mul, (x, y) -> x * y);
        return out(s, o);
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return mul(a, 1.0f / scalar);
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 2);
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.binary(a, b, o.segment(), n, CpuOps::div, (x, y) -> x / y);
        return out(s, o);
    }

    private Tensor broadcastOp(Tensor a, Tensor b, int op) {
        // Simple naive broadcasting for CPU fallback
        TensorOperationValidator.BroadcastPlan plan =
                TensorOperationValidator.validateBroadcast(a.shape(), b.shape());
        long[] shapeA = a.shape().dims();
        long[] shapeB = b.shape().dims();
        long[] outDims = plan.outputDimensions();
        int maxRank = outDims.length;
        long[] stridesA = new long[maxRank];
        long[] stridesB = new long[maxRank];
        long numel = plan.outputElements();

        long sa = 1, sb = 1;
        for (int i = 0; i < maxRank; i++) {
            int aIdx = shapeA.length - 1 - i;
            int bIdx = shapeB.length - 1 - i;
            long dimA = aIdx >= 0 ? shapeA[aIdx] : 1;
            long dimB = bIdx >= 0 ? shapeB[bIdx] : 1;
            stridesA[maxRank - 1 - i] = dimA == 1 ? 0 : sa;
            stridesB[maxRank - 1 - i] = dimB == 1 ? 0 : sb;
            if (aIdx >= 0)
                sa *= shapeA[aIdx];
            if (bIdx >= 0)
                sb *= shapeB[bIdx];
        }

        HostAccessibleStorage o = allocateOutput(numel * 4);
        MemorySegment segA = seg(a);
        MemorySegment segB = seg(b);
        MemorySegment segO = o.segment();

        for (long i = 0; i < numel; i++) {
            long flatA = 0;
            long flatB = 0;
            long remaining = i;
            for (int d = 0; d < maxRank; d++) {
                long div = 1;
                for (int j = d + 1; j < maxRank; j++)
                    div *= outDims[j];
                long idx = remaining / div;
                remaining %= div;
                flatA += idx * stridesA[d];
                flatB += idx * stridesB[d];
            }
            float valA = segA.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, flatA * 4);
            float valB = segB.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, flatB * 4);
            float res = 0;
            if (op == 0)
                res = valA + valB;
            else if (op == 1)
                res = valA * valB;
            else if (op == 2)
                res = valA / valB;
            else if (op == 3)
                res = valA - valB;
            segO.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, i * 4, res);
        }
        return out(new Shape(outDims), o);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, (src, dst, count) -> CpuOps.addScalar(src, scalar, dst, count), x -> x + scalar);
        return out(s, o);
    }

    // ── Matmul ────────────────────────────────────────────────────────────────

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        long[] shapeA = a.shape().dims();
        long[] shapeB = b.shape().dims();
        if (shapeA.length < 2 || shapeB.length < 2) {
            throw new IllegalArgumentException("matmul requires at least 2D tensors");
        }
        if (a.dtype() != DType.F32 || b.dtype() != DType.F32) {
            throw new UnsupportedOperationException("CPU matmul currently supports F32 tensors only");
        }

        TensorOperationValidator.MatmulPlan plan =
                TensorOperationValidator.validateMatmul(a.shape(), b.shape());
        int batchRankA = shapeA.length - 2;
        int batchRankB = shapeB.length - 2;
        int batchRank = plan.batchDimensions().length;
        int outputRank = batchRank + 2;
        int m = Math.toIntExact(plan.rows());
        int k = Math.toIntExact(plan.inner());
        int n = Math.toIntExact(plan.columns());
        long[] outputBatchDims = plan.batchDimensions();
        long[] outDims = plan.outputDimensions();

        Shape outShape = new Shape(outDims);
        HostAccessibleStorage outBuf = allocateOutput(
                Math.multiplyExact(outShape.numel(), (long) Float.BYTES));
        MemorySegment output = outBuf.segment();
        output.fill((byte) 0);

        long[] stridesA = batchStrides(shapeA);
        long[] stridesB = batchStrides(shapeB);
        long batchCount = 1;
        for (long dimension : outputBatchDims) {
            batchCount = Math.multiplyExact(batchCount, dimension);
        }
        long leftMatrixBytes = Math.multiplyExact(Math.multiplyExact((long) m, k), Float.BYTES);
        long rightMatrixBytes = Math.multiplyExact(Math.multiplyExact((long) k, n), Float.BYTES);
        long outputMatrixBytes = Math.multiplyExact(Math.multiplyExact((long) m, n), Float.BYTES);
        MemorySegment left = seg(a);
        MemorySegment right = seg(b);

        for (long batch = 0; batch < batchCount; batch++) {
            long remaining = batch;
            long leftElementOffset = 0;
            long rightElementOffset = 0;
            for (int axis = batchRank - 1; axis >= 0; axis--) {
                long dimension = outputBatchDims[axis];
                long coordinate = remaining % dimension;
                remaining /= dimension;
                int axisA = axis - (batchRank - batchRankA);
                int axisB = axis - (batchRank - batchRankB);
                if (axisA >= 0 && shapeA[axisA] != 1) {
                    leftElementOffset = Math.addExact(leftElementOffset,
                            Math.multiplyExact(coordinate, stridesA[axisA]));
                }
                if (axisB >= 0 && shapeB[axisB] != 1) {
                    rightElementOffset = Math.addExact(rightElementOffset,
                            Math.multiplyExact(coordinate, stridesB[axisB]));
                }
            }

            long outputOffset = Math.multiplyExact(batch, outputMatrixBytes);
            MatmulCpu.matmul(
                    left.asSlice(Math.multiplyExact(leftElementOffset, Float.BYTES), leftMatrixBytes),
                    right.asSlice(Math.multiplyExact(rightElementOffset, Float.BYTES), rightMatrixBytes),
                    output.asSlice(outputOffset, outputMatrixBytes),
                    m, n, k);
        }
        return out(outShape, outBuf);
    }

    /** Returns row-major batch strides in elements, excluding the final matrix dimensions. */
    private static long[] batchStrides(long[] shape) {
        int batchRank = shape.length - 2;
        long[] strides = new long[batchRank];
        long stride = Math.multiplyExact(shape[batchRank], shape[batchRank + 1]);
        for (int axis = batchRank - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, shape[axis]);
        }
        return strides;
    }

    // ── Shape ops ────────────────────────────────────────────────────────────

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        Shape outputShape = new Shape(newShape);
        if (outputShape.numel() != a.numel()) {
            throw new IllegalArgumentException(
                    "reshape must preserve the number of elements: input="
                            + a.numel() + ", output=" + outputShape.numel());
        }

        TensorLayout inputLayout = a.layout();
        if (!inputLayout.isContiguous()) {
            throw new UnsupportedOperationException(
                    "CPU reshape of a non-contiguous tensor requires materialization; "
                            + "input layout=" + inputLayout);
        }

        // A contiguous tensor may still be a view with a non-zero storage
        // offset. Preserve that offset while assigning the output's canonical
        // row-major strides.
        TensorLayout contiguousOutput = TensorLayout.contiguous(outputShape);
        TensorLayout outputLayout = new TensorLayout(
                outputShape, contiguousOutput.strides(), inputLayout.storageOffset());
        TensorStorage storage = a.storage();
        storage.retain();
        try {
            return new DefaultTensor(
                    outputShape, a.dtype(), a.device(), storage, this, outputLayout);
        } catch (RuntimeException | Error e) {
            storage.release();
            throw e;
        }
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        Shape inShape = a.shape();
        int rank = inShape.rank();
        if (rank == 0) {
            throw new IllegalArgumentException("Cannot slice a scalar tensor");
        }
        if (offsets == null || sizes == null || offsets.length != rank || sizes.length != rank) {
            throw new IllegalArgumentException(
                    "Slice offsets and sizes must both have one entry per tensor dimension");
        }
        if (a.dtype() != DType.F32) {
            throw new UnsupportedOperationException(
                    "CPU slice currently supports F32 tensors only; actual dtype is " + a.dtype());
        }

        Shape outShape = new Shape(sizes);
        long[] inDims = inShape.dims();
        TensorLayout inputLayout = a.layout();
        long[] inputStrides = inputLayout.strides();
        long outputOffset = inputLayout.storageOffset();
        for (int i = 0; i < rank; i++) {
            if (offsets[i] < 0 || offsets[i] > inDims[i]
                    || sizes[i] > inDims[i] - offsets[i]) {
                throw new IllegalArgumentException(
                        "Slice is outside tensor bounds at dimension " + i
                                + ": offset=" + offsets[i] + ", size=" + sizes[i]
                                + ", dimension=" + inDims[i]);
            }
            outputOffset = Math.addExact(
                    outputOffset, Math.multiplyExact(offsets[i], inputStrides[i]));
        }

        // A slice is a logical sub-view of the input: preserve its physical
        // strides and advance the storage offset by the requested coordinates.
        // This works for contiguous, offset, and already-strided input tensors.
        TensorLayout outputLayout = new TensorLayout(outShape, inputStrides, outputOffset);
        hostStorage(a); // Reject device-only storage at the CPU boundary.
        TensorStorage storage = a.storage();
        storage.retain();
        try {
            return new DefaultTensor(
                    outShape, a.dtype(), a.device(), storage, this, outputLayout);
        } catch (RuntimeException | Error failure) {
            storage.release();
            throw failure;
        }
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        Objects.requireNonNull(a, "input");
        TensorOperationValidator.SplitPlan plan =
                TensorOperationValidator.validateSplit(a.shape(), axis, parts);
        int normalizedAxis = plan.axis();
        long[] dims = a.shape().dims();
        long chunkSize = plan.chunkSize();
        List<Tensor> result = new ArrayList<>(parts);
        long[] offsets = new long[rank];
        long[] sizes = dims.clone();
        sizes[normalizedAxis] = chunkSize;
        try {
            for (int p = 0; p < parts; p++) {
                offsets[normalizedAxis] = Math.multiplyExact((long) p, chunkSize);
                result.add(slice(a, offsets.clone(), sizes.clone()));
            }
            return result;
        } catch (RuntimeException | Error failure) {
            for (Tensor tensor : result) {
                tensor.close();
            }
            throw failure;
        }
    }

    @Override
    public Tensor flatten(Tensor a) {
        Objects.requireNonNull(a, "input");
        Shape outputShape = TensorOperationValidator.validateFlatten(a.shape());
        return reshape(a, outputShape.dims());
    }

    @Override
    public Tensor unsqueeze(Tensor a, int dim) {
        TensorOperationValidator.UnsqueezePlan plan =
                TensorOperationValidator.validateUnsqueeze(a.shape(), dim);
        return reshape(a, plan.outputShape().dims());
    }

    @Override
    public Tensor squeeze(Tensor a) {
        Shape outputShape = TensorOperationValidator.validateSqueeze(a.shape());
        return reshape(a, outputShape.dims());
    }

    @Override
    public Tensor transpose(Tensor a) {
        Objects.requireNonNull(a, "input");
        TensorOperationValidator.TransposePlan plan =
                TensorOperationValidator.validateTranspose(a.shape());
        return transpose(a, plan.axis0(), plan.axis1());
    }

    @Override
    public Tensor transpose(Tensor a, int d0, int d1) {
        Objects.requireNonNull(a, "input");
        TensorOperationValidator.TransposePlan plan =
                TensorOperationValidator.validateTranspose(a.shape(), d0, d1);
        d0 = plan.axis0();
        d1 = plan.axis1();
        if (a.dtype() != DType.F32) {
            throw new UnsupportedOperationException(
                    "CPU transpose currently supports F32 tensors only; actual dtype is " + a.dtype());
        }

        int rank = a.shape().rank();
        Shape outShape = plan.outputShape();
        long[] outDims = outShape.dims();
        long total = outShape.numel();
        long outputBytes = Math.multiplyExact(total, Float.BYTES);
        HostAccessibleStorage outBuf = allocateOutput(outputBytes);
        try {
            TensorLayout inputLayout = a.layout();
            long[] inputStrides = inputLayout.strides();
            long storageOffset = inputLayout.storageOffset();
            long[] outputStrides = TensorLayout.contiguous(outShape).strides();
            MemorySegment src = seg(a);
            MemorySegment dst = outBuf.segment();
            long[] cursor = new long[rank];

            for (long element = 0; element < total; element++) {
                long inputIndex = storageOffset;
                long outputIndex = 0;
                for (int dimension = 0; dimension < rank; dimension++) {
                    int inputDimension = dimension == d0 ? d1
                            : dimension == d1 ? d0 : dimension;
                    inputIndex = Math.addExact(inputIndex,
                            Math.multiplyExact(cursor[dimension], inputStrides[inputDimension]));
                    outputIndex = Math.addExact(outputIndex,
                            Math.multiplyExact(cursor[dimension], outputStrides[dimension]));
                }
                long inputByteOffset = Math.multiplyExact(inputIndex, Float.BYTES);
                long outputByteOffset = Math.multiplyExact(outputIndex, Float.BYTES);
                dst.set(ValueLayout.JAVA_FLOAT, outputByteOffset,
                        src.get(ValueLayout.JAVA_FLOAT, inputByteOffset));

                for (int dimension = rank - 1; dimension >= 0; dimension--) {
                    if (++cursor[dimension] < outDims[dimension]) {
                        break;
                    }
                    cursor[dimension] = 0;
                }
            }
            return out(outShape, outBuf);
        } catch (RuntimeException | Error failure) {
            outBuf.release();
            throw failure;
        }
    }

    // ── Activations ───────────────────────────────────────────────────────────

    @Override
    public Tensor relu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::relu, x -> Math.max(0f, x));
        return out(s, o);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::sigmoid, x -> (float) (1d / (1d + Math.exp(-x))));
        return out(s, o);
    }

    @Override
    public Tensor tanh(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::tanh, x -> (float) Math.tanh(x));
        return out(s, o);
    }

    @Override
    public Tensor log(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::log, x -> (float) Math.log(x));
        return out(s, o);
    }

    @Override
    public Tensor exp(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::exp, x -> (float) Math.exp(x));
        return out(s, o);
    }

    @Override
    public Tensor silu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::silu, x -> x / (1f + (float) Math.exp(-x)));
        return out(s, o);
    }

    @Override
    public Tensor gelu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuTensorLayoutAccess.unary(a, o.segment(), n, CpuOps::gelu, x -> 0.5f * x * (1f + (float) Math.tanh(0.7978845608f * (x + 0.044715f * x * x * x))));
        return out(s, o);
    }

    // ── Normalization ─────────────────────────────────────────────────────────

    @Override
    public Tensor softmax(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        return softmax(a, a.shape().rank() - 1);
    }

    @Override
    public Tensor softmax(Tensor a, int dim) {
        return normalizeAlongSuffix(a, dim, false);
    }

    @Override
    public Tensor logSoftmax(Tensor a, int dim) {
        return normalizeAlongSuffix(a, dim, true);
    }

    /**
     * Computes softmax or log-softmax over the flattened suffix beginning at {@code dim}.
     * This preserves the existing CPU backend axis convention while gathering inputs
     * through their logical tensor layout.
     */
    private Tensor normalizeAlongSuffix(Tensor a, int dim, boolean logarithmic) {
        Objects.requireNonNull(a, "tensor must not be null");
        CpuTensorLayoutAccess.requireF32(a);
        Shape shape = a.shape();
        long[] dimensions = shape.dims();
        int rank = dimensions.length;
        if (rank == 0) {
            throw new IllegalArgumentException("Normalization axis requires a tensor with rank greater than zero");
        }
        int normalizedDim = dim < 0 ? rank + dim : dim;
        if (normalizedDim < 0 || normalizedDim >= rank) {
            throw new IllegalArgumentException(
                    "Normalization axis " + dim + " is out of range for tensor rank " + rank);
        }

        long rowLength = 1;
        for (int axis = normalizedDim; axis < rank; axis++) {
            rowLength = Math.multiplyExact(rowLength, dimensions[axis]);
        }
        long rowCount = 1;
        for (int axis = 0; axis < normalizedDim; axis++) {
            rowCount = Math.multiplyExact(rowCount, dimensions[axis]);
        }

        long elements = shape.numel();
        HostAccessibleStorage output = allocateOutput(
                Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) {
            return out(shape, output);
        }

        MemorySegment source = seg(a);
        MemorySegment destination = output.segment();
        TensorLayout layout = a.layout();
        if (!logarithmic && layout.isContiguous() && layout.storageOffset() == 0
                && rowLength <= Integer.MAX_VALUE) {
            int contiguousRowLength = Math.toIntExact(rowLength);
            for (long row = 0; row < rowCount; row++) {
                CpuOps.softmax(source, destination, Math.multiplyExact(row, rowLength),
                        contiguousRowLength);
            }
            return out(shape, output);
        }

        for (long row = 0; row < rowCount; row++) {
            long rowStart = Math.multiplyExact(row, rowLength);
            float max = Float.NEGATIVE_INFINITY;
            for (long column = 0; column < rowLength; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                max = Math.max(max, CpuTensorLayoutAccess.readLogical(
                        source, dimensions, layout, logicalIndex));
            }

            double exponentialSum = 0d;
            for (long column = 0; column < rowLength; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                float value = CpuTensorLayoutAccess.readLogical(
                        source, dimensions, layout, logicalIndex);
                exponentialSum += Math.exp((double) value - max);
            }
            double logNormalizer = Math.log(exponentialSum);
            float inverseSum = (float) (1d / exponentialSum);
            for (long column = 0; column < rowLength; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                float value = CpuTensorLayoutAccess.readLogical(
                        source, dimensions, layout, logicalIndex);
                float result = logarithmic
                        ? (float) ((double) value - max - logNormalizer)
                        : (float) (Math.exp((double) value - max) * inverseSum);
                destination.set(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(logicalIndex, Float.BYTES), result);
            }
        }
        return out(shape, output);
    }

    @Override
    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        CpuTensorLayoutAccess.requireF32(input);
        requireNormalizationEpsilon(eps);
        Shape shape = input.shape();
        long[] dimensions = shape.dims();
        if (normalizedShape.length == 0 || normalizedShape.length > dimensions.length) {
            throw new IllegalArgumentException("normalizedShape must be a non-empty suffix of the input shape");
        }
        long hiddenDim = 1;
        for (int i = 0; i < normalizedShape.length; i++) {
            long expected = normalizedShape[i];
            long actual = dimensions[dimensions.length - normalizedShape.length + i];
            if (expected <= 0 || expected != actual) {
                throw new IllegalArgumentException("normalizedShape must match the trailing input dimensions");
            }
            hiddenDim = Math.multiplyExact(hiddenDim, expected);
        }
        validateNormParameter(weight, hiddenDim, "weight");
        validateNormParameter(bias, hiddenDim, "bias");
        long elements = shape.numel();
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) return out(shape, output);

        MemorySegment source = seg(input), destination = output.segment();
        MemorySegment scale = weight == null ? null : seg(weight);
        MemorySegment shift = bias == null ? null : seg(bias);
        TensorLayout layout = input.layout();
        long rows = elements / hiddenDim;
        for (long row = 0; row < rows; row++) {
            long rowStart = Math.multiplyExact(row, hiddenDim);
            double sum = 0d;
            for (long column = 0; column < hiddenDim; column++) {
                sum += CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                        Math.addExact(rowStart, column));
            }
            double mean = sum / hiddenDim;
            double squaredDifferenceSum = 0d;
            for (long column = 0; column < hiddenDim; column++) {
                double difference = CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                        Math.addExact(rowStart, column)) - mean;
                squaredDifferenceSum += difference * difference;
            }
            float inverseStd = (float) (1d / Math.sqrt(squaredDifferenceSum / hiddenDim + eps));
            for (long column = 0; column < hiddenDim; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                float normalized = (value - (float) mean) * inverseStd;
                if (scale != null) normalized *= scale.get(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(column, Float.BYTES));
                if (shift != null) normalized += shift.get(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(column, Float.BYTES));
                destination.set(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(logicalIndex, Float.BYTES), normalized);
            }
        }
        return out(shape, output);
    }

    @Override
    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        CpuTensorLayoutAccess.requireF32(input);
        CpuTensorLayoutAccess.requireF32(weight);
        requireNormalizationEpsilon(eps);
        if (weight.shape().rank() != 1) {
            throw new IllegalArgumentException("RMSNorm weight must be a rank-1 vector");
        }
        long hiddenDim = weight.shape().dim(0);
        if (hiddenDim <= 0 || input.shape().rank() == 0
                || input.shape().dim(input.shape().rank() - 1) != hiddenDim) {
            throw new IllegalArgumentException("RMSNorm weight length must match the input's final dimension");
        }
        validateNormParameter(weight, hiddenDim, "weight");
        long elements = input.numel();
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) return out(input.shape(), output);

        MemorySegment source = seg(input), scale = seg(weight), destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();
        long rows = elements / hiddenDim;
        for (long row = 0; row < rows; row++) {
            long rowStart = Math.multiplyExact(row, hiddenDim);
            double sumSquares = 0d;
            for (long column = 0; column < hiddenDim; column++) {
                float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                        Math.addExact(rowStart, column));
                sumSquares += (double) value * value;
            }
            float inverseRms = (float) (1d / Math.sqrt(sumSquares / hiddenDim + eps));
            for (long column = 0; column < hiddenDim; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                float gamma = scale.get(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(column, Float.BYTES));
                destination.set(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(logicalIndex, Float.BYTES), (value * inverseRms) * gamma);
            }
        }
        return out(input.shape(), output);
    }

    private static void validateNormParameter(Tensor parameter, long expectedElements, String name) {
        if (parameter == null) return;
        CpuTensorLayoutAccess.requireF32(parameter);
        if (parameter.numel() != expectedElements) {
            throw new IllegalArgumentException(name + " must contain " + expectedElements
                    + " elements; got " + parameter.numel());
        }
        TensorLayout layout = parameter.layout();
        if (!layout.isContiguous() || layout.storageOffset() != 0) {
            throw new IllegalArgumentException(name + " must use contiguous zero-offset storage");
        }
    }

    private static void requireNormalizationEpsilon(float eps) {
        if (!Float.isFinite(eps) || eps < 0f) {
            throw new IllegalArgumentException("Normalization epsilon must be finite and non-negative");
        }
    }

    @Override
    public Tensor applyRoPE(Tensor input, float freqBase, int rotaryDimension,
            boolean isNeox, float positionScale, long[] positionIds) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(positionIds, "positionIds");
        if (input.backend() != this) throw new IllegalArgumentException("RoPE input must belong to this CPU backend instance");
        if (input.dtype() != DType.F32) throw new UnsupportedOperationException("CPU RoPE currently supports F32 tensors only");
        if (input.shape().rank() != 4) throw new IllegalArgumentException("RoPE expects [batch, heads, sequence, headDim]");
        if (!Float.isFinite(freqBase) || freqBase <= 1f
                || !Float.isFinite(positionScale) || positionScale <= 0f) {
            throw new IllegalArgumentException("Invalid RoPE frequency base or position scale");
        }
        long batch=input.shape().dim(0), heads=input.shape().dim(1), sequence=input.shape().dim(2);
        int headDim=Math.toIntExact(input.shape().dim(3));
        if (positionIds.length != Math.multiplyExact(batch, sequence)) {
            throw new IllegalArgumentException("positionIds length must equal batch * sequence");
        }
        if (batch<=0 || heads<=0 || sequence<=0 || rotaryDimension<=0
                || (rotaryDimension&1)!=0 || rotaryDimension>headDim) {
            throw new IllegalArgumentException("rotaryDimension must be positive, even, and <= headDim");
        }
        for (long position : positionIds) if (position < 0) {
            throw new IllegalArgumentException("position IDs must be non-negative");
        }
        long elements=input.numel();
        long byteSize=Math.multiplyExact(elements, Float.BYTES);
        HostAccessibleStorage output=allocateOutput(byteSize);
        MemorySegment source=seg(input), destination=output.segment();
        destination.copyFrom(source.asSlice(0, byteSize));
        int half=rotaryDimension/2, sequenceInt=Math.toIntExact(sequence);
        for(long b=0;b<batch;b++) for(long h=0;h<heads;h++) for(int s=0;s<sequenceInt;s++) {
            double pos=positionIds[Math.toIntExact(b)*sequenceInt+s]/positionScale;
            long row=((b*heads+h)*sequence+s)*headDim;
            for(int pair=0;pair<half;pair++) {
                double angle=pos*Math.pow(freqBase,-2.0*pair/rotaryDimension);
                float cos=(float)Math.cos(angle), sin=(float)Math.sin(angle);
                int first=isNeox?pair:pair*2, second=isNeox?pair+half:pair*2+1;
                long fo=(row+first)*Float.BYTES, so=(row+second)*Float.BYTES;
                float x=source.get(ValueLayout.JAVA_FLOAT,fo), y=source.get(ValueLayout.JAVA_FLOAT,so);
                destination.set(ValueLayout.JAVA_FLOAT,fo,x*cos-y*sin);
                destination.set(ValueLayout.JAVA_FLOAT,so,x*sin+y*cos);
            }
        }
        return out(input.shape(),output);
    }

    @Override
    public Tensor applyRoPE(Tensor input, int posOffset, float freqBase,
            int rotaryDimension, boolean isNeox, float positionScale) {
        Objects.requireNonNull(input, "input");
        if (input.backend() != this) throw new IllegalArgumentException("RoPE input must belong to this CPU backend instance");
        if (input.dtype() != DType.F32) throw new UnsupportedOperationException("CPU RoPE currently supports F32 tensors only");
        if (input.shape().rank() != 4) throw new IllegalArgumentException("RoPE expects [batch, heads, sequence, headDim]");
        if (posOffset < 0 || !Float.isFinite(freqBase) || freqBase <= 1f
                || !Float.isFinite(positionScale) || positionScale <= 0f) {
            throw new IllegalArgumentException("Invalid RoPE offset, frequency base, or position scale");
        }
        long batch=input.shape().dim(0), heads=input.shape().dim(1), sequence=input.shape().dim(2);
        int headDim=Math.toIntExact(input.shape().dim(3));
        if (batch<=0 || heads<=0 || sequence<=0 || rotaryDimension<=0
                || (rotaryDimension&1)!=0 || rotaryDimension>headDim) {
            throw new IllegalArgumentException("rotaryDimension must be positive, even, and <= headDim");
        }
        long elements=input.numel();
        HostAccessibleStorage output=allocateOutput(Math.multiplyExact(elements, Float.BYTES));
        MemorySegment source=seg(input), destination=output.segment();
        destination.copyFrom(source.asSlice(0, Math.multiplyExact(elements, Float.BYTES)));
        int half=rotaryDimension/2;
        for(long b=0;b<batch;b++) for(long h=0;h<heads;h++) for(long s=0;s<sequence;s++) {
            double pos=(posOffset+s)/positionScale;
            long row=((b*heads+h)*sequence+s)*headDim;
            for(int pair=0;pair<half;pair++) {
                double angle=pos*Math.pow(freqBase,-2.0*pair/rotaryDimension);
                float cos=(float)Math.cos(angle), sin=(float)Math.sin(angle);
                int first=isNeox?pair:pair*2, second=isNeox?pair+half:pair*2+1;
                long fo=(row+first)*Float.BYTES, so=(row+second)*Float.BYTES;
                float x=source.get(ValueLayout.JAVA_FLOAT,fo), y=source.get(ValueLayout.JAVA_FLOAT,so);
                destination.set(ValueLayout.JAVA_FLOAT,fo,x*cos-y*sin);
                destination.set(ValueLayout.JAVA_FLOAT,so,x*sin+y*cos);
            }
        }
        return out(input.shape(),output);
    }

    @Override
    public Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        Objects.requireNonNull(input, "input");
        if (input.backend() != this) {
            throw new IllegalArgumentException("RoPE input must belong to this CPU backend instance");
        }
        if (input.dtype() != DType.F32) {
            throw new UnsupportedOperationException("CPU RoPE currently supports F32 tensors only");
        }
        if (input.shape().rank() != 4) {
            throw new IllegalArgumentException("RoPE expects [batch, heads, sequence, headDim]");
        }
        if (posOffset < 0 || !Float.isFinite(freqBase) || freqBase <= 1.0f) {
            throw new IllegalArgumentException("posOffset must be non-negative and freqBase must be finite and > 1");
        }

        long batch = input.shape().dim(0);
        long heads = input.shape().dim(1);
        long sequence = input.shape().dim(2);
        int headDim = Math.toIntExact(input.shape().dim(3));
        if (batch <= 0 || heads <= 0 || sequence <= 0 || headDim <= 0 || (headDim & 1) != 0) {
            throw new IllegalArgumentException("RoPE requires positive dimensions and an even headDim");
        }

        long elements = input.numel();
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(elements, Float.BYTES));
        MemorySegment source = seg(input);
        MemorySegment destination = output.segment();
        destination.copyFrom(source.asSlice(0, Math.multiplyExact(elements, Float.BYTES)));

        int halfDim = headDim / 2;
        for (long b = 0; b < batch; b++) {
            for (long h = 0; h < heads; h++) {
                for (long s = 0; s < sequence; s++) {
                    long position = Math.addExact(posOffset, s);
                    long row = ((b * heads + h) * sequence + s) * headDim;
                    for (int pair = 0; pair < halfDim; pair++) {
                        double exponent = -2.0 * pair / headDim;
                        double angle = position * Math.pow(freqBase, exponent);
                        float cos = (float) Math.cos(angle);
                        float sin = (float) Math.sin(angle);
                        int first = isNeox ? pair : pair * 2;
                        int second = isNeox ? pair + halfDim : pair * 2 + 1;
                        long firstOffset = (row + first) * Float.BYTES;
                        long secondOffset = (row + second) * Float.BYTES;
                        float x = source.get(ValueLayout.JAVA_FLOAT, firstOffset);
                        float y = source.get(ValueLayout.JAVA_FLOAT, secondOffset);
                        destination.set(ValueLayout.JAVA_FLOAT, firstOffset, x * cos - y * sin);
                        destination.set(ValueLayout.JAVA_FLOAT, secondOffset, x * sin + y * cos);
                    }
                }
            }
        }
        return out(input.shape(), output);
    }

    @Override
    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias,
            Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        Objects.requireNonNull(input, "input");
        CpuTensorLayoutAccess.requireF32(input);
        if (input.shape().rank() < 2) {
            throw new IllegalArgumentException("BatchNorm input must have rank >= 2 ([N, C, ...])");
        }
        if (!Float.isFinite(momentum) || momentum < 0f || momentum > 1f) {
            throw new IllegalArgumentException("BatchNorm momentum must be finite and in [0, 1]");
        }
        if (!Float.isFinite(eps) || eps < 0f) {
            throw new IllegalArgumentException("BatchNorm epsilon must be finite and non-negative");
        }

        long[] dimensions = input.shape().dims();
        long batches = dimensions[0];
        long channels = dimensions[1];
        if (channels <= 0) {
            throw new IllegalArgumentException("BatchNorm channel dimension must be positive");
        }
        long spatialSize = 1;
        for (int i = 2; i < dimensions.length; i++) {
            spatialSize = Math.multiplyExact(spatialSize, dimensions[i]);
        }
        long samplesPerChannel = Math.multiplyExact(batches, spatialSize);
        validateBatchNormVector(weight, channels, "weight");
        validateBatchNormVector(bias, channels, "bias");
        if ((runningMean == null) != (runningVar == null)) {
            throw new IllegalArgumentException("runningMean and runningVar must either both be present or both be null");
        }
        validateBatchNormVector(runningMean, channels, "runningMean");
        validateBatchNormVector(runningVar, channels, "runningVar");
        long elements = input.numel();
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) return out(input.shape(), output);

        MemorySegment source = seg(input);
        MemorySegment destination = output.segment();
        MemorySegment scale = weight == null ? null : seg(weight);
        MemorySegment shift = bias == null ? null : seg(bias);
        MemorySegment means = runningMean == null ? null : seg(runningMean);
        MemorySegment variances = runningVar == null ? null : seg(runningVar);
        TensorLayout layout = input.layout();

        for (long channel = 0; channel < channels; channel++) {
            double mean;
            double variance;
            if (training || means == null) {
                double sum = 0d;
                double sumSquares = 0d;
                for (long batch = 0; batch < batches; batch++) {
                    long base = Math.multiplyExact(
                            Math.addExact(Math.multiplyExact(batch, channels), channel), spatialSize);
                    for (long spatial = 0; spatial < spatialSize; spatial++) {
                        float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                                Math.addExact(base, spatial));
                        sum += value;
                        sumSquares += (double) value * value;
                    }
                }
                mean = sum / samplesPerChannel;
                variance = Math.max(0d, sumSquares / samplesPerChannel - mean * mean);

                if (training && means != null) {
                    double runningVariance = samplesPerChannel > 1
                            ? variance * samplesPerChannel / (samplesPerChannel - 1d)
                            : variance;
                    long parameterOffset = Math.multiplyExact(channel, Float.BYTES);
                    float oldMean = means.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                    float oldVariance = variances.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                    if (!Float.isFinite(oldMean) || !Float.isFinite(oldVariance) || oldVariance < 0f) {
                        throw new IllegalArgumentException(
                                "BatchNorm running statistics must be finite and variance non-negative");
                    }
                    means.set(ValueLayout.JAVA_FLOAT, parameterOffset,
                            (float) ((1d - momentum) * oldMean + momentum * mean));
                    variances.set(ValueLayout.JAVA_FLOAT, parameterOffset,
                            (float) ((1d - momentum) * oldVariance + momentum * runningVariance));
                }
            } else {
                long parameterOffset = Math.multiplyExact(channel, Float.BYTES);
                mean = means.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                variance = variances.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                if (!Double.isFinite(mean) || !Double.isFinite(variance) || variance < 0d) {
                    throw new IllegalArgumentException("BatchNorm running statistics must be finite and variance non-negative");
                }
            }

            float inverseStd = (float) (1d / Math.sqrt(variance + eps));
            float gamma = scale == null ? 1f
                    : scale.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(channel, Float.BYTES));
            float beta = shift == null ? 0f
                    : shift.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(channel, Float.BYTES));
            for (long batch = 0; batch < batches; batch++) {
                long base = Math.multiplyExact(
                        Math.addExact(Math.multiplyExact(batch, channels), channel), spatialSize);
                for (long spatial = 0; spatial < spatialSize; spatial++) {
                    long logicalIndex = Math.addExact(base, spatial);
                    float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                    float normalized = (float) ((value - mean) * inverseStd) * gamma + beta;
                    destination.set(ValueLayout.JAVA_FLOAT,
                            Math.multiplyExact(logicalIndex, Float.BYTES), normalized);
                }
            }
        }
        return out(input.shape(), output);
    }

    private static void validateBatchNormVector(Tensor tensor, long channels, String name) {
        if (tensor == null) return;
        CpuTensorLayoutAccess.requireF32(tensor);
        if (tensor.shape().rank() != 1 || tensor.shape().dim(0) != channels) {
            throw new IllegalArgumentException(name + " must be a rank-1 vector of length " + channels);
        }
        TensorLayout layout = tensor.layout();
        if (!layout.isContiguous() || layout.storageOffset() != 0) {
            throw new IllegalArgumentException(name + " must use contiguous zero-offset storage");
        }
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    @Override
    public Tensor sum(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        CpuTensorLayoutAccess.requireF32(a);
        MemorySegment source = seg(a);
        TensorLayout layout = a.layout();
        float total;
        if (layout.isContiguous() && layout.storageOffset() == 0) {
            total = CpuOps.reduceSum(source, a.numel());
        } else {
            total = 0f;
            long[] dimensions = a.shape().dims();
            for (long i = 0; i < a.numel(); i++) {
                total += CpuTensorLayoutAccess.readLogical(source, dimensions, layout, i);
            }
        }
        HostAccessibleStorage o = allocateOutput(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, total);
        return out(new Shape(1), o);
    }

    @Override
    public Tensor sum(Tensor a, int dim, boolean keepDim) {
        return reduceAlongDim(a, dim, keepDim, true);
    }

    @Override
    public Tensor mean(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        if (a.numel() == 0) {
            throw new IllegalArgumentException("Mean is undefined for an empty tensor");
        }
        CpuTensorLayoutAccess.requireF32(a);
        MemorySegment source = seg(a);
        TensorLayout layout = a.layout();
        float total;
        if (layout.isContiguous() && layout.storageOffset() == 0) {
            total = CpuOps.reduceSum(source, a.numel());
        } else {
            total = 0f;
            long[] dimensions = a.shape().dims();
            for (long i = 0; i < a.numel(); i++) {
                total += CpuTensorLayoutAccess.readLogical(source, dimensions, layout, i);
            }
        }
        HostAccessibleStorage o = allocateOutput(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, total / a.numel());
        return out(new Shape(1), o);
    }

    @Override
    public Tensor mean(Tensor a, int dim, boolean keepDim) {
        return reduceAlongDim(a, dim, keepDim, false);
    }

    @Override
    public Tensor max(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        CpuTensorLayoutAccess.requireF32(a);
        MemorySegment source = seg(a);
        TensorLayout layout = a.layout();
        float best;
        if (layout.isContiguous() && layout.storageOffset() == 0) {
            best = CpuOps.reduceMax(source, a.numel());
        } else {
            best = Float.NEGATIVE_INFINITY;
            long[] dimensions = a.shape().dims();
            for (long i = 0; i < a.numel(); i++) {
                best = Math.max(best,
                        CpuTensorLayoutAccess.readLogical(source, dimensions, layout, i));
            }
        }
        HostAccessibleStorage o = allocateOutput(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, best);
        return out(new Shape(1), o);
    }

    /** Shared reduce-along-dim logic for sum and mean. */
    private Tensor reduceAlongDim(Tensor a, int dim, boolean keepDim, boolean isSum) {
        Objects.requireNonNull(a, "tensor must not be null");
        CpuTensorLayoutAccess.requireF32(a);
        long[] inDims = a.shape().dims();
        int rank = inDims.length;
        if (rank == 0) {
            throw new IllegalArgumentException("Axis reduction requires a tensor with rank greater than zero");
        }
        int normalizedDim = dim < 0 ? rank + dim : dim;
        if (normalizedDim < 0 || normalizedDim >= rank) {
            throw new IllegalArgumentException(
                    "Reduction axis " + dim + " is out of range for tensor rank " + rank);
        }
        dim = normalizedDim;
        long dimLen = inDims[dim];
        if (!isSum && dimLen == 0) {
            throw new IllegalArgumentException("Mean is undefined for an empty reduction dimension");
        }

        long[] outDims = new long[keepDim ? rank : rank - 1];
        int outIdx = 0;
        for (int d = 0; d < rank; d++) {
            if (d == dim && !keepDim) {
                continue;
            }
            outDims[outIdx++] = (d == dim) ? 1 : inDims[d];
        }
        Shape outShape = new Shape(outDims);
        HostAccessibleStorage outBuf = allocateOutput(
                Math.multiplyExact(outShape.numel(), (long) Float.BYTES));
        MemorySegment source = seg(a);
        MemorySegment destination = outBuf.segment();
        TensorLayout layout = a.layout();

        long outerSize = 1;
        long innerSize = 1;
        for (int d = 0; d < dim; d++) {
            outerSize = Math.multiplyExact(outerSize, inDims[d]);
        }
        for (int d = dim + 1; d < rank; d++) {
            innerSize = Math.multiplyExact(innerSize, inDims[d]);
        }

        for (long outer = 0; outer < outerSize; outer++) {
            for (long inner = 0; inner < innerSize; inner++) {
                float accumulator = 0f;
                for (long reduced = 0; reduced < dimLen; reduced++) {
                    long logicalInputIndex = Math.addExact(
                            Math.multiplyExact(Math.addExact(
                                    Math.multiplyExact(outer, dimLen), reduced), innerSize), inner);
                    accumulator += CpuTensorLayoutAccess.readLogical(
                            source, inDims, layout, logicalInputIndex);
                }
                if (!isSum) {
                    accumulator /= dimLen;
                }
                long outputIndex = Math.addExact(
                        Math.multiplyExact(outer, innerSize), inner);
                destination.set(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(outputIndex, Float.BYTES), accumulator);
            }
        }
        return out(outShape, outBuf);
    }

    // ── Math ─────────────────────────────────────────────────────────────────

    @Override
    public Tensor pow(Tensor a, float exponent) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++)
            CpuOps.setF(dst, i, (float) Math.pow(CpuOps.getF(src, i), exponent));
        return out(s, o);
    }

    @Override
    public Tensor sign(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * Float.BYTES);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++) {
            CpuOps.setF(dst, i, Math.signum(CpuOps.getF(src, i)));
        }
        return out(s, o);
    }

    @Override
    public Tensor abs(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++)
            CpuOps.setF(dst, i, Math.abs(CpuOps.getF(src, i)));
        return out(s, o);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++)
            CpuOps.setF(dst, i, (float) Math.sqrt(CpuOps.getF(src, i)));
        return out(s, o);
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        Shape s = a.shape();
        HostAccessibleStorage o = allocateOutput(s.numel() * 4);
        o.segment().fill((byte) 0);
        return out(s, o);
    }

    @Override
    public Tensor cast(Tensor a, DType dtype) {
        if (a.dtype() == dtype)
            return a;

        long n = a.numel();
        HostAccessibleStorage outBuf = allocateOutput(dtype.memoryFootprintBytes(n));

        if (a.dtype() == DType.F32 && dtype == DType.Q8_0) {
            tech.kayys.aljabr.backend.cpu.ops.QuantizeOps.quantizeQ8_0(seg(a), outBuf.segment(), n);
        } else if (a.dtype() == DType.F32 && dtype == DType.Q4_0) {
            tech.kayys.aljabr.backend.cpu.ops.QuantizeOps.quantizeQ4_0(seg(a), outBuf.segment(), n);
        } else if (a.dtype() == DType.Q8_0 && dtype == DType.F32) {
            tech.kayys.aljabr.backend.cpu.ops.DequantizeOps.dequantizeQ8_0(seg(a), outBuf.segment(), n);
        } else if (a.dtype() == DType.Q4_0 && dtype == DType.F32) {
            tech.kayys.aljabr.backend.cpu.ops.DequantizeOps.dequantizeQ4_0(seg(a), outBuf.segment(), n);
        } else {
            throw new UnsupportedOperationException("cast from " + a.dtype() + " to " + dtype + " not supported yet");
        }

        return new DefaultTensor(a.shape(), dtype, a.device(), outBuf, this);
    }

    @Override
    public Tensor to(Tensor a, DeviceType device) {
        if (a.device() == device)
            return a;
        throw new UnsupportedOperationException("Cross-device copy not supported in CpuBackend");
    }

    // ── Dropout ──────────────────────────────────────────────────────────────

    @Override
    public Tensor dropout(Tensor input, float p, boolean training) {
        if (!training || p == 0f)
            return input;
        Shape s = input.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.dropout(seg(input), o.segment(), n, p, ThreadLocalRandom.current().nextLong());
        return out(s, o);
    }

    // ── Loss ─────────────────────────────────────────────────────────────────

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) {
        // pred: [N, C] log-probabilities (after log_softmax); target: [N] class indices (I32)
        Objects.requireNonNull(pred, "pred");
        Objects.requireNonNull(target, "target");
        if (pred.dtype() != DType.F32 || target.dtype() != DType.I32) {
            throw new UnsupportedOperationException("crossEntropy requires F32 predictions and I32 class indices");
        }
        if (pred.shape().rank() != 2 || target.shape().rank() != 1) {
            throw new IllegalArgumentException("crossEntropy requires predictions [N, C] and targets [N]");
        }
        long batchSize = pred.shape().dim(0);
        long numClasses = pred.shape().dim(1);
        if (batchSize <= 0 || numClasses <= 0 || target.shape().dim(0) != batchSize) {
            throw new IllegalArgumentException("crossEntropy requires non-empty [N, C] predictions and matching [N] targets");
        }
        MemorySegment sp = seg(pred);
        MemorySegment st = seg(target);
        float loss = 0f;
        for (long i = 0; i < batchSize; i++) {
            int classIdx = st.get(ValueLayout.JAVA_INT, i * Integer.BYTES);
            if (classIdx < 0 || classIdx >= numClasses) {
                throw new IllegalArgumentException(
                        "crossEntropy target class index " + classIdx + " is outside [0, " + numClasses + ")");
            }
            loss -= sp.get(ValueLayout.JAVA_FLOAT, (i * numClasses + classIdx) * Float.BYTES);
        }
        loss /= batchSize;
        HostAccessibleStorage o = allocateOutput(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, loss);
        return out(new Shape(1), o);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        // pred: predicted probabilities; target: same-shaped binary labels in {0,1}
        Objects.requireNonNull(pred, "pred");
        Objects.requireNonNull(target, "target");
        if (pred.dtype() != DType.F32 || target.dtype() != DType.F32) {
            throw new UnsupportedOperationException("binaryCrossEntropy requires F32 predictions and targets");
        }
        if (pred.shape().rank() == 0 || !sameShape(pred.shape(), target.shape())) {
            throw new IllegalArgumentException("binaryCrossEntropy requires non-scalar predictions and matching target shape");
        }
        long n = pred.numel();
        if (n <= 0) {
            throw new IllegalArgumentException("binaryCrossEntropy requires non-empty tensors");
        }
        MemorySegment sp = seg(pred);
        MemorySegment st = seg(target);
        float loss = 0f;
        for (long i = 0; i < n; i++) {
            float p = Math.max(1e-7f, Math.min(1 - 1e-7f, CpuOps.getF(sp, i)));
            float t = CpuOps.getF(st, i);
            if (!Float.isFinite(t) || (t != 0f && t != 1f)) {
                throw new IllegalArgumentException("binaryCrossEntropy targets must contain only finite 0 or 1 labels");
            }
            loss -= t * (float) Math.log(p) + (1 - t) * (float) Math.log(1 - p);
        }
        loss /= n;
        HostAccessibleStorage o = allocateOutput(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, loss);
        return out(new Shape(1), o);
    }

    private static boolean sameShape(Shape left, Shape right) {
        if (left.rank() != right.rank()) {
            return false;
        }
        for (int i = 0; i < left.rank(); i++) {
            if (left.dim(i) != right.dim(i)) {
                return false;
            }
        }
        return true;
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        return attention(Q, K, V, null, false);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V, Tensor additiveMask, boolean causal) {
        if (Q == null || K == null || V == null) throw new IllegalArgumentException("attention Q, K, and V must not be null");
        if (Q.backend() != this || K.backend() != this || V.backend() != this)
            throw new IllegalArgumentException("attention tensors must belong to this CPU backend instance");
        if (Q.dtype() != DType.F32 || K.dtype() != DType.F32 || V.dtype() != DType.F32)
            throw new UnsupportedOperationException("CPU attention currently supports F32 tensors only");
        int rank = Q.shape().rank();
        if (rank == 2 && additiveMask == null && !causal) {
            if (K.shape().rank() != 2 || V.shape().rank() != 2
                    || Q.shape().dim(0) != K.shape().dim(0) || K.shape().dim(0) != V.shape().dim(0)
                    || Q.shape().dim(1) != K.shape().dim(1) || K.shape().dim(1) != V.shape().dim(1))
                throw new IllegalArgumentException("rank-2 attention expects matching [sequence, headDim] Q/K/V");
            return FlashAttentionCpu.forward(Q, K, V, Runtime.getRuntime().availableProcessors());
        }
        if (rank != 4 || K.shape().rank() != 4 || V.shape().rank() != 4)
            throw new IllegalArgumentException("masked attention expects rank-4 [B,H,S,D] Q/K/V tensors");
        long batch=Q.shape().dim(0), heads=Q.shape().dim(1), queryLength=Q.shape().dim(2), headDim=Q.shape().dim(3), keyLength=K.shape().dim(2);
        if (batch<=0 || heads<=0 || queryLength<=0 || keyLength<=0 || headDim<=0
                || K.shape().dim(0)!=batch || K.shape().dim(1)!=heads || K.shape().dim(3)!=headDim
                || V.shape().dim(0)!=batch || V.shape().dim(1)!=heads || V.shape().dim(2)!=keyLength || V.shape().dim(3)!=headDim)
            throw new IllegalArgumentException("rank-4 attention expects Q[B,H,Sq,D], K/V[B,H,Sk,D] with matching B/H/D");
        if (additiveMask != null && (additiveMask.backend()!=this || additiveMask.dtype()!=DType.F32
                || additiveMask.shape().rank()!=2 || additiveMask.shape().dim(0)!=queryLength
                || additiveMask.shape().dim(1)!=keyLength))
            throw new IllegalArgumentException("additive attention mask must be F32 [queryLength, keyLength] on this backend");
        MemorySegment mask = additiveMask == null ? null : seg(additiveMask);
        if (mask != null) {
            validateAttentionMask(mask, queryLength, keyLength);
        }
        Shape outputShape=new Shape(batch,heads,queryLength,headDim);
        HostAccessibleStorage output=allocateOutput(Math.multiplyExact(outputShape.numel(),Float.BYTES));
        MemorySegment q=seg(Q), k=seg(K), v=seg(V), out=output.segment();
        float scale=(float)(1.0/Math.sqrt(headDim));
        int groupCount = Math.toIntExact(Math.multiplyExact(batch, heads));
        try {
            if (shouldParallelizeAttention(groupCount, queryLength, keyLength, headDim)) {
                // The common ForkJoinPool bounds CPU attention workers globally
                // across backend instances and concurrent inference requests.
                java.util.stream.IntStream.range(0, groupCount).parallel().forEach(group ->
                        computeAttentionGroup(q, k, v, mask, out, group, group,
                                queryLength, keyLength, headDim, scale, causal));
            } else {
                for (int group = 0; group < groupCount; group++) {
                    computeAttentionGroup(q, k, v, mask, out, group, group,
                            queryLength, keyLength, headDim, scale, causal);
                }
            }
            return out(outputShape, output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }


    @Override
    public Tensor attentionGqa(Tensor Q, Tensor K, Tensor V, Tensor additiveMask, boolean causal) {
        if (Q == null || K == null || V == null) {
            throw new IllegalArgumentException("attention Q, K, and V must not be null");
        }
        if (Q.backend() != this || K.backend() != this || V.backend() != this) {
            throw new IllegalArgumentException("attention tensors must belong to this CPU backend instance");
        }
        if (Q.dtype() != DType.F32 || K.dtype() != DType.F32 || V.dtype() != DType.F32) {
            throw new UnsupportedOperationException("CPU attention currently supports F32 tensors only");
        }
        if (Q.shape().rank() != 4 || K.shape().rank() != 4 || V.shape().rank() != 4) {
            throw new IllegalArgumentException("GQA expects rank-4 [B,H,S,D] Q/K/V tensors");
        }

        long batch = Q.shape().dim(0);
        long queryHeads = Q.shape().dim(1);
        long queryLength = Q.shape().dim(2);
        long headDim = Q.shape().dim(3);
        long keyValueHeads = K.shape().dim(1);
        long keyLength = K.shape().dim(2);

        if (batch <= 0 || queryHeads <= 0 || keyValueHeads <= 0
                || queryLength <= 0 || keyLength <= 0 || headDim <= 0
                || queryHeads % keyValueHeads != 0
                || K.shape().dim(0) != batch || V.shape().dim(0) != batch
                || K.shape().dim(3) != headDim || V.shape().dim(1) != keyValueHeads
                || V.shape().dim(2) != keyLength || V.shape().dim(3) != headDim) {
            throw new IllegalArgumentException(
                    "GQA expects Q[B,Hq,Sq,D], K/V[B,Hkv,Sk,D] with Hq divisible by Hkv");
        }

        if (additiveMask != null && (additiveMask.backend() != this
                || additiveMask.dtype() != DType.F32
                || additiveMask.shape().rank() != 2
                || additiveMask.shape().dim(0) != queryLength
                || additiveMask.shape().dim(1) != keyLength)) {
            throw new IllegalArgumentException(
                    "GQA additive attention mask must be F32 [queryLength, keyLength] on this backend");
        }

        MemorySegment mask = additiveMask == null ? null : seg(additiveMask);
        if (mask != null) validateAttentionMask(mask, queryLength, keyLength);

        Shape outputShape = new Shape(batch, queryHeads, queryLength, headDim);
        HostAccessibleStorage output =
                allocateOutput(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment q = seg(Q), k = seg(K), v = seg(V), out = output.segment();
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int groupCount = Math.toIntExact(Math.multiplyExact(batch, queryHeads));
        int queriesPerKeyValueHead = Math.toIntExact(queryHeads / keyValueHeads);
        try {
            if (shouldParallelizeAttention(groupCount, queryLength, keyLength, headDim)) {
                java.util.stream.IntStream.range(0, groupCount).parallel().forEach(group -> {
                    int queryHead = group % Math.toIntExact(queryHeads);
                    int keyValueHead = queryHead / queriesPerKeyValueHead;
                    computeAttentionGroup(q, k, v, mask, out, group,
                            (group / Math.toIntExact(queryHeads))
                                    * Math.toIntExact(keyValueHeads) + keyValueHead,
                            queryLength, keyLength, headDim, scale, causal);
                });
            } else {
                for (int group = 0; group < groupCount; group++) {
                    int queryHead = group % Math.toIntExact(queryHeads);
                    int keyValueHead = queryHead / queriesPerKeyValueHead;
                    computeAttentionGroup(q, k, v, mask, out, group,
                            (group / Math.toIntExact(queryHeads))
                                    * Math.toIntExact(keyValueHeads) + keyValueHead,
                            queryLength, keyLength, headDim, scale, causal);
                }
            }
            return out(outputShape, output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }

    /**
     * Validates a shared additive mask once before attention work is scheduled.
     * The same mask is reused by every batch/head group, so validation must not
     * be repeated in each group's query/key hot loop.
     */
    private static void validateAttentionMask(
            MemorySegment mask, long queryLength, long keyLength) {
        long elements = Math.multiplyExact(queryLength, keyLength);
        for (long index = 0; index < elements; index++) {
            float value = mask.get(ValueLayout.JAVA_FLOAT, index * Float.BYTES);
            if (Float.isNaN(value) || value == Float.POSITIVE_INFINITY) {
                throw new IllegalArgumentException(
                        "attention mask values must not be NaN or positive infinity");
            }
        }
    }

    /**
     * Avoid parallel scheduling for small attention workloads where task
     * coordination costs more than processing independent batch/head groups.
     */
    private static boolean shouldParallelizeAttention(
            int groupCount, long queryLength, long keyLength, long headDim) {
        if (groupCount < 2) {
            return false;
        }
        long work = groupCount;
        if (work > Long.MAX_VALUE / queryLength) {
            return true;
        }
        work *= queryLength;
        if (work > Long.MAX_VALUE / keyLength) {
            return true;
        }
        work *= keyLength;
        if (work > Long.MAX_VALUE / headDim) {
            return true;
        }
        work *= headDim;
        return work >= (1L << 18);
    }

    /** Computes one independent [batch, head] attention group. */
    private static void computeAttentionGroup(
            MemorySegment q,
            MemorySegment k,
            MemorySegment v,
            MemorySegment mask,
            MemorySegment out,
            int group,
            int keyGroup,
            long queryLength,
            long keyLength,
            long headDim,
            float scale,
            boolean causal) {
        int dimension = Math.toIntExact(headDim);
        long qBase = (long) group * queryLength * dimension;
        long kBase = (long) keyGroup * keyLength * dimension;
        double[] weightedValues = new double[dimension];

        for (long qi = 0; qi < queryLength; qi++) {
            java.util.Arrays.fill(weightedValues, 0d);
            float runningMax = Float.NEGATIVE_INFINITY;
            double denominator = 0d;
            long causalLimit = keyLength - queryLength + qi;
            long maskRowOffset = qi * keyLength * Float.BYTES;

            // Online softmax: keep only the running normalization and weighted
            // value sum; no query-sized score/probability array is allocated.
            // Bound the loop directly so causally excluded keys incur neither
            // an inner-loop branch nor an additive-mask memory read.
            long keyLimit = causal ? Math.min(keyLength, causalLimit + 1) : keyLength;
            for (long ki = 0; ki < keyLimit; ki++) {
                float additive = mask == null ? 0f
                        : mask.get(ValueLayout.JAVA_FLOAT, maskRowOffset + ki * Float.BYTES);
                if (additive == Float.NEGATIVE_INFINITY) {
                    continue;
                }

                float score = CpuAttentionOps.dot(
                        q, (qBase + qi * dimension) * Float.BYTES,
                        k, (kBase + ki * dimension) * Float.BYTES,
                        dimension) * scale + additive;

                float nextMax;
                double previousScale;
                double currentScale;
                if (Float.isFinite(score)) {
                    if (score > runningMax) {
                        previousScale = Math.exp(runningMax - score);
                        currentScale = 1d;
                        nextMax = score;
                    } else {
                        previousScale = 1d;
                        currentScale = Math.exp(score - runningMax);
                        nextMax = runningMax;
                    }
                } else {
                    // Preserve the existing exceptional-value behavior.
                    nextMax = Math.max(runningMax, score);
                    previousScale = runningMax == Float.NEGATIVE_INFINITY
                            ? 0d : Math.exp(runningMax - nextMax);
                    currentScale = Math.exp(score - nextMax);
                }
                denominator = denominator * previousScale + currentScale;

                for (int d = 0; d < dimension; d++) {
                    float valueAtKey = v.get(ValueLayout.JAVA_FLOAT,
                            (kBase + ki * dimension + d) * Float.BYTES);
                    weightedValues[d] = weightedValues[d] * previousScale
                            + currentScale * valueAtKey;
                }
                runningMax = nextMax;
            }

            for (int d = 0; d < dimension; d++) {
                float result = denominator == 0d
                        ? 0f : (float) (weightedValues[d] / denominator);
                out.set(ValueLayout.JAVA_FLOAT,
                        (qBase + qi * dimension + d) * Float.BYTES, result);
            }
        }
    }

    // ── Conv / Pool ───────────────────────────────────────────────────────────

    @Override
    public Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
            int stride, int padding, int dilation, int groups) {
        // Im2col + GEMM approach
        long N = input.shape().dim(0);
        long Cin = input.shape().dim(1);
        long H = input.shape().dim(2);
        long W = input.shape().dim(3);
        long Cout = weight.shape().dim(0);
        long kH = weight.shape().dim(2);
        long kW = weight.shape().dim(3);

        long outH = (H + 2L * padding - dilation * (kH - 1) - 1) / stride + 1;
        long outW = (W + 2L * padding - dilation * (kW - 1) - 1) / stride + 1;

        Shape outShape = new Shape(N, Cout, outH, outW);
        HostAccessibleStorage outBuf = allocateOutput(outShape.numel() * 4);
        outBuf.segment().fill((byte) 0);

        MemorySegment si = seg(input);
        MemorySegment sw = seg(weight);
        MemorySegment so = outBuf.segment();

        for (long n = 0; n < N; n++) {
            for (long oc = 0; oc < Cout; oc++) {
                float biasVal = (bias != null)
                        ? seg(bias).get(ValueLayout.JAVA_FLOAT, oc * 4L)
                        : 0f;
                for (long oh = 0; oh < outH; oh++) {
                    for (long ow = 0; ow < outW; ow++) {
                        float acc = biasVal;
                        for (long ic = 0; ic < Cin; ic++) {
                            for (long kh = 0; kh < kH; kh++) {
                                for (long kw = 0; kw < kW; kw++) {
                                    long ih = oh * stride - padding + kh * dilation;
                                    long iw = ow * stride - padding + kw * dilation;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                        continue;
                                    float inVal = si.get(ValueLayout.JAVA_FLOAT,
                                            ((n * Cin + ic) * H + ih) * W * 4L + iw * 4L);
                                    float wVal = sw.get(ValueLayout.JAVA_FLOAT,
                                            ((oc * Cin + ic) * kH + kh) * kW * 4L + kw * 4L);
                                    acc += inVal * wVal;
                                }
                            }
                        }
                        so.set(ValueLayout.JAVA_FLOAT,
                                ((n * Cout + oc) * outH + oh) * outW * 4L + ow * 4L, acc);
                    }
                }
            }
        }
        return out(outShape, outBuf);
    }

    @Override
    public Tensor upsampleNearest2d(Tensor input, int outputH, int outputW) {
        requireSpatialInput(input, "upsampleNearest2d");
        if (outputH <= 0 || outputW <= 0) {
            throw new IllegalArgumentException("output height and width must be positive");
        }
        long batch = input.shape().dim(0);
        long channels = input.shape().dim(1);
        long inputH = input.shape().dim(2);
        long inputW = input.shape().dim(3);
        if (inputH <= 0 || inputW <= 0) {
            throw new IllegalArgumentException("input height and width must be positive");
        }

        Shape outputShape = new Shape(batch, channels, outputH, outputW);
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment source = seg(input);
        MemorySegment destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();

        for (long n = 0; n < batch; n++) {
            for (long c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputH; oh++) {
                    long ih = Math.min(inputH - 1, (long) oh * inputH / outputH);
                    for (int ow = 0; ow < outputW; ow++) {
                        long iw = Math.min(inputW - 1, (long) ow * inputW / outputW);
                        long logicalIndex = ((n * channels + c) * inputH + ih) * inputW + iw;
                        long destinationIndex = ((n * channels + c) * outputH + oh) * outputW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, destinationIndex * Float.BYTES,
                                CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex));
                    }
                }
            }
        }
        return out(outputShape, output);
    }

    @Override
    public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        requireSpatialInput(input, "maxPool2d");
        if (kernelSize <= 0 || stride <= 0 || padding < 0) {
            throw new IllegalArgumentException("kernel size and stride must be positive; padding must be non-negative");
        }
        long nSize = input.shape().dim(0);
        long channels = input.shape().dim(1);
        long height = input.shape().dim(2);
        long width = input.shape().dim(3);
        long outH = (height + 2L * padding - kernelSize) / stride + 1;
        long outW = (width + 2L * padding - kernelSize) / stride + 1;
        if (outH <= 0 || outW <= 0) {
            throw new IllegalArgumentException("pooling kernel produces an empty spatial output");
        }

        Shape outputShape = new Shape(nSize, channels, outH, outW);
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment source = seg(input);
        MemorySegment destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();

        for (long n = 0; n < nSize; n++) {
            for (long c = 0; c < channels; c++) {
                for (long oh = 0; oh < outH; oh++) {
                    for (long ow = 0; ow < outW; ow++) {
                        float best = Float.NEGATIVE_INFINITY;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                long ih = oh * stride - padding + kh;
                                long iw = ow * stride - padding + kw;
                                if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;
                                long logicalIndex = ((n * channels + c) * height + ih) * width + iw;
                                float value = CpuTensorLayoutAccess.readLogical(
                                        source, dimensions, layout, logicalIndex);
                                if (value > best) best = value;
                            }
                        }
                        long outputIndex = ((n * channels + c) * outH + oh) * outW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, outputIndex * Float.BYTES, best);
                    }
                }
            }
        }
        return out(outputShape, output);
    }

    @Override
    public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        requireSpatialInput(input, "adaptiveAvgPool2d");
        if (outputH <= 0 || outputW <= 0) {
            throw new IllegalArgumentException("output height and width must be positive");
        }
        long nSize = input.shape().dim(0);
        long channels = input.shape().dim(1);
        long height = input.shape().dim(2);
        long width = input.shape().dim(3);
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("input height and width must be positive");
        }

        Shape outputShape = new Shape(nSize, channels, outputH, outputW);
        HostAccessibleStorage output = allocateOutput(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment source = seg(input);
        MemorySegment destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();

        for (long n = 0; n < nSize; n++) {
            for (long c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputH; oh++) {
                    for (int ow = 0; ow < outputW; ow++) {
                        long hStart = (long) oh * height / outputH;
                        long hEnd = (long) (oh + 1) * height / outputH;
                        long wStart = (long) ow * width / outputW;
                        long wEnd = (long) (ow + 1) * width / outputW;
                        double sum = 0d;
                        long count = 0;
                        for (long ih = hStart; ih < hEnd; ih++) {
                            for (long iw = wStart; iw < wEnd; iw++) {
                                long logicalIndex = ((n * channels + c) * height + ih) * width + iw;
                                sum += CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                                count++;
                            }
                        }
                        long outputIndex = ((n * channels + c) * outputH + oh) * outputW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, outputIndex * Float.BYTES,
                                count == 0 ? 0f : (float) (sum / count));
                    }
                }
            }
        }
        return out(outputShape, output);
    }

    private static void requireSpatialInput(Tensor input, String operation) {
        Objects.requireNonNull(input, "input");
        if (input.shape().rank() != 4) {
            throw new IllegalArgumentException(operation + " expects NCHW rank-4 input");
        }
        CpuTensorLayoutAccess.requireF32(input);
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    @Override
    public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
        long[] inputDims = input.shape().dims();
        long vocabSize = weight.shape().dim(0);
        long embeddingDim = weight.shape().dim(1);

        long[] outputDims = new long[inputDims.length + 1];
        System.arraycopy(inputDims, 0, outputDims, 0, inputDims.length);
        outputDims[inputDims.length] = embeddingDim;

        long numTokens = input.numel();
        HostAccessibleStorage outBuf = allocateOutput(numTokens * embeddingDim * 4);
        MemorySegment sw = seg(weight);
        MemorySegment si = seg(input);
        MemorySegment so = outBuf.segment();

        DType inputDType = input.dtype();
        for (long i = 0; i < numTokens; i++) {
            long idx = switch (inputDType) {
                case I32 -> si.get(ValueLayout.JAVA_INT, i * 4L);
                case I8 -> si.get(ValueLayout.JAVA_BYTE, i);
                default -> (long) CpuOps.getF(si, i);
            };
            long outOff = i * embeddingDim * 4L;
            if (idx == paddingIdx || idx < 0 || idx >= vocabSize) {
                so.asSlice(outOff, embeddingDim * 4L).fill((byte) 0);
            } else {
                MemorySegment.copy(sw, idx * embeddingDim * 4L, so, outOff, embeddingDim * 4L);
            }
        }
        return new DefaultTensor(new Shape(outputDims), DType.F32, DeviceType.CPU, outBuf, this);
    }

    // ── Misc ──────────────────────────────────────────────────────────────────

    @Override
    public long numel(Tensor a) {
        return a.numel();
    }

    @Override
    public Tensor cat(List<Tensor> tensors, int axis) {
        TensorOperationValidator.CatPlan plan =
                TensorOperationValidator.validateCat(tensors, axis, DiagnosticSink.noop());
        if (plan.first().dtype() != DType.F32) {
            throw new UnsupportedOperationException("CPU cat currently supports F32 tensors only");
        }
        long[] outputDims = plan.outputDimensions();
        int normalizedAxis = plan.axis();
        Tensor first = plan.first();
        int rank = first.shape().rank();
        long outer = 1, inner = 1;
        for (int d = 0; d < normalizedAxis; d++) outer = Math.multiplyExact(outer, first.shape().dim(d));
        for (int d = normalizedAxis + 1; d < rank; d++) inner = Math.multiplyExact(inner, first.shape().dim(d));
        float[][] sources = new float[tensors.size()][];
        for (int i = 0; i < tensors.size(); i++) sources[i] = tensors.get(i).toFloatArray();
        long outputElements = new Shape(outputDims).numel();
        if (outputElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("cat result is too large for a Java float[]: " + outputElements);
        }
        float[] output = new float[Math.toIntExact(outputElements)];
        int destination = 0;
        for (long outerIndex = 0; outerIndex < outer; outerIndex++) {
            for (int i = 0; i < tensors.size(); i++) {
                long blockElements = Math.multiplyExact(tensors.get(i).shape().dim(normalizedAxis), inner);
                int blockLength = Math.toIntExact(blockElements);
                int sourceOffset = Math.toIntExact(Math.multiplyExact(outerIndex, blockElements));
                System.arraycopy(sources[i], sourceOffset, output, destination, blockLength);
                destination += blockLength;
            }
        }
        return TensorFactory.of(output, this, outputDims);
    }


}
