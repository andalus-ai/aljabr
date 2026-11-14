package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.backend.cpu.ops.CpuOps;
import tech.kayys.aljabr.backend.cpu.ops.MatmulCpu;
import tech.kayys.aljabr.backend.cpu.ops.NormOps;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
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
        CpuOps.add(seg(a), seg(b), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 3);
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.sub(seg(a), seg(b), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.mulScalar(seg(a), scalar, o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 1);
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.mul(seg(a), seg(b), o.segment(), n);
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
        CpuOps.div(seg(a), seg(b), o.segment(), n);
        return out(s, o);
    }

    private Tensor broadcastOp(Tensor a, Tensor b, int op) {
        // Simple naive broadcasting for CPU fallback
        long[] shapeA = a.shape().dims();
        long[] shapeB = b.shape().dims();
        int maxRank = Math.max(shapeA.length, shapeB.length);
        long[] outDims = new long[maxRank];
        long[] stridesA = new long[maxRank];
        long[] stridesB = new long[maxRank];

        long numel = 1;
        for (int i = 0; i < maxRank; i++) {
            int aIdx = shapeA.length - 1 - i;
            int bIdx = shapeB.length - 1 - i;
            long dimA = aIdx >= 0 ? shapeA[aIdx] : 1;
            long dimB = bIdx >= 0 ? shapeB[bIdx] : 1;
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new UnsupportedOperationException("Shapes not broadcastable: " + a.shape() + " and " + b.shape());
            }
            outDims[maxRank - 1 - i] = Math.max(dimA, dimB);
            numel *= outDims[maxRank - 1 - i];
        }

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
        CpuOps.addScalar(seg(a), scalar, o.segment(), n);
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

        int batchRankA = shapeA.length - 2;
        int batchRankB = shapeB.length - 2;
        int batchRank = Math.max(batchRankA, batchRankB);
        int outputRank = batchRank + 2;
        int m = Math.toIntExact(shapeA[shapeA.length - 2]);
        int k = Math.toIntExact(shapeA[shapeA.length - 1]);
        int k2 = Math.toIntExact(shapeB[shapeB.length - 2]);
        int n = Math.toIntExact(shapeB[shapeB.length - 1]);
        if (k != k2) {
            throw new IllegalArgumentException("Dimension mismatch: " + k + " != " + k2);
        }

        long[] outputBatchDims = new long[batchRank];
        for (int axis = 0; axis < batchRank; axis++) {
            int axisA = axis - (batchRank - batchRankA);
            int axisB = axis - (batchRank - batchRankB);
            long dimA = axisA < 0 ? 1 : shapeA[axisA];
            long dimB = axisB < 0 ? 1 : shapeB[axisB];
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException(
                        "matmul batch dimensions are not broadcast-compatible at aligned axis "
                                + axis + ": " + dimA + " != " + dimB);
            }
            outputBatchDims[axis] = Math.max(dimA, dimB);
        }

        long[] outDims = java.util.Arrays.copyOf(outputBatchDims, outputRank);
        outDims[outputRank - 2] = m;
        outDims[outputRank - 1] = n;
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
        long stride = 1;
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

        // Share the same backend-neutral storage and give the result its own
        // ownership reference. This does not require the legacy Buffer API.
        TensorStorage storage = a.storage();
        storage.retain();
        try {
            return new DefaultTensor(outputShape, a.dtype(), a.device(), storage, this);
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
        for (int i = 0; i < rank; i++) {
            if (offsets[i] < 0 || offsets[i] > inDims[i]
                    || sizes[i] > inDims[i] - offsets[i]) {
                throw new IllegalArgumentException(
                        "Slice is outside tensor bounds at dimension " + i
                                + ": offset=" + offsets[i] + ", size=" + sizes[i]
                                + ", dimension=" + inDims[i]);
            }
        }

        // Strides in elements (row-major) for the input.
        long[] baseStrides = new long[rank];
        baseStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            baseStrides[i] = Math.multiplyExact(baseStrides[i + 1], inDims[i + 1]);
        }

        // A rectangular slice is contiguous when its dimensions preserve the
        // input's trailing row-major strides.
        long[] viewExpected = new long[rank];
        long expectedStride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            viewExpected[i] = expectedStride;
            expectedStride = Math.multiplyExact(expectedStride, sizes[i]);
        }

        boolean isContiguous = true;
        for (int i = 0; i < rank; i++) {
            if (baseStrides[i] != viewExpected[i]) {
                isContiguous = false;
                break;
            }
        }

        HostAccessibleStorage parentStorage = hostStorage(a);
        MemorySegment src = parentStorage.segment();
        long elemOffset = 0;
        for (int i = 0; i < rank; i++) {
            elemOffset = Math.addExact(elemOffset, Math.multiplyExact(offsets[i], baseStrides[i]));
        }
        long byteOffset = Math.multiplyExact(elemOffset, Float.BYTES);
        long byteSize = Math.multiplyExact(outShape.numel(), Float.BYTES);

        if (isContiguous) {
            // Preserve the legacy Buffer view type for existing callers, while
            // allowing non-Buffer host storage to use a neutral zero-copy view.
            TensorStorage viewStorage = parentStorage instanceof Buffer legacyBuffer
                    ? CpuBuffer.view(legacyBuffer, byteOffset, byteSize)
                    : HostAccessibleStorageView.view(parentStorage, byteOffset, byteSize);
            try {
                return new DefaultTensor(outShape, a.dtype(), a.device(), viewStorage, this);
            } catch (RuntimeException | Error e) {
                viewStorage.release();
                throw e;
            }
        }

        // Non-contiguous slices are materialized into an independent CPU buffer.
        HostAccessibleStorage outBuf = allocateOutput(byteSize);
        long outIdx = 0;
        long totalOut = outShape.numel();
        long[] cursor = new long[rank];
        for (long elem = 0; elem < totalOut; elem++) {
            long inIdx = 0;
            for (int d = 0; d < rank; d++) {
                inIdx = Math.addExact(inIdx,
                        Math.multiplyExact(Math.addExact(offsets[d], cursor[d]), baseStrides[d]));
            }
            outBuf.segment().set(ValueLayout.JAVA_FLOAT, outIdx * Float.BYTES,
                    src.get(ValueLayout.JAVA_FLOAT, inIdx * Float.BYTES));
            outIdx++;
            for (int d = rank - 1; d >= 0; d--) {
                if (++cursor[d] < sizes[d]) {
                    break;
                }
                cursor[d] = 0;
            }
        }
        return out(outShape, outBuf);
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        long[] dims = a.shape().dims();
        long totalOnAxis = dims[axis];
        long chunkSize = totalOnAxis / parts;
        List<Tensor> result = new ArrayList<>(parts);
        long[] offsets = new long[dims.length];
        long[] sizes = dims.clone();
        sizes[axis] = chunkSize;
        for (int p = 0; p < parts; p++) {
            offsets[axis] = p * chunkSize;
            result.add(slice(a, offsets.clone(), sizes.clone()));
        }
        return result;
    }

    @Override
    public Tensor flatten(Tensor a) {
        return reshape(a, a.numel());
    }

    @Override
    public Tensor unsqueeze(Tensor a, int dim) {
        long[] dims = a.shape().dims();
        int rank = dims.length;
        if (dim < 0)
            dim += rank + 1;
        long[] newDims = new long[rank + 1];
        System.arraycopy(dims, 0, newDims, 0, dim);
        newDims[dim] = 1;
        System.arraycopy(dims, dim, newDims, dim + 1, rank - dim);
        return reshape(a, newDims);
    }

    @Override
    public Tensor squeeze(Tensor a) {
        long[] dims = a.shape().dims();
        long[] newDims = java.util.Arrays.stream(dims).filter(d -> d != 1).toArray();
        if (newDims.length == 0)
            newDims = new long[] { 1 };
        return reshape(a, newDims);
    }

    @Override
    public Tensor transpose(Tensor a) {
        long[] dims = a.shape().dims();
        int rank = dims.length;
        return transpose(a, rank - 2, rank - 1);
    }

    @Override
    public Tensor transpose(Tensor a, int d0, int d1) {
        long[] inDims = a.shape().dims();
        int rank = inDims.length;
        long[] outDims = inDims.clone();
        outDims[d0] = inDims[d1];
        outDims[d1] = inDims[d0];

        Shape outShape = new Shape(outDims);
        HostAccessibleStorage outBuf = allocateOutput(outShape.numel() * 4);

        // Compute strides for both input and output (row-major)
        long[] inStrides = new long[rank];
        long[] outStrides = new long[rank];
        inStrides[rank - 1] = outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            inStrides[i] = inStrides[i + 1] * inDims[i + 1];
            outStrides[i] = outStrides[i + 1] * outDims[i + 1];
        }

        MemorySegment src = seg(a);
        MemorySegment dst = outBuf.segment();
        long total = outShape.numel();
        long[] cursor = new long[rank];
        for (long elem = 0; elem < total; elem++) {
            long inIdx = 0, outIdx = 0;
            for (int d = 0; d < rank; d++) {
                int inD = (d == d0) ? d1 : (d == d1) ? d0 : d;
                inIdx += cursor[d] * inStrides[inD];
                outIdx += cursor[d] * outStrides[d];
            }
            long memOffset = inIdx * 4L;
            if (memOffset >= src.byteSize()) {
                System.err.println("OOB Access! inDims=" + java.util.Arrays.toString(inDims) + ", d0=" + d0 + ", d1="
                        + d1 + ", outDims=" + java.util.Arrays.toString(outDims) + ", inStrides="
                        + java.util.Arrays.toString(inStrides) + ", outStrides=" + java.util.Arrays.toString(outStrides)
                        + ", cursor=" + java.util.Arrays.toString(cursor) + ", inIdx=" + inIdx + ", memOffset="
                        + memOffset + ", byteSize=" + src.byteSize());
            }
            dst.set(ValueLayout.JAVA_FLOAT, outIdx * 4L,
                    src.get(ValueLayout.JAVA_FLOAT, memOffset));
            for (int d = rank - 1; d >= 0; d--) {
                if (++cursor[d] < outDims[d])
                    break;
                cursor[d] = 0;
            }
        }
        return out(outShape, outBuf);
    }

    // ── Activations ───────────────────────────────────────────────────────────

    @Override
    public Tensor relu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.relu(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.sigmoid(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor tanh(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.tanh(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor log(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.log(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor exp(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.exp(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor silu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.silu(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor gelu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        HostAccessibleStorage o = allocateOutput(n * 4);
        CpuOps.gelu(seg(a), o.segment(), n);
        return out(s, o);
    }

    // ── Normalization ─────────────────────────────────────────────────────────

    @Override
    public Tensor softmax(Tensor a) {
        return softmax(a, a.shape().rank() - 1);
    }

    @Override
    public Tensor softmax(Tensor a, int dim) {
        Shape shape = a.shape();
        long n = shape.numel();
        HostAccessibleStorage outBuf = allocateOutput(n * 4);
        MemorySegment src = seg(a);

        // Flatten to 2D: (batchRows, rowLen) where softmax is over rowLen
        int rowLen = 1;
        long[] dims = shape.dims();
        for (int d = dim; d < dims.length; d++)
            rowLen *= (int) dims[d];
        long numRows = n / rowLen;

        for (long r = 0; r < numRows; r++)
            CpuOps.softmax(src, outBuf.segment(), r * rowLen, rowLen);

        return out(shape, outBuf);
    }

    @Override
    public Tensor logSoftmax(Tensor a, int dim) {
        // log(softmax(x)) = x - log(sum(exp(x))) computed stably as:
        // log_softmax(x) = x - max - log(sum(exp(x - max)))
        Tensor sm = softmax(a, dim);
        return log(sm);
    }

    @Override
    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        Shape shape = input.shape();
        long n = shape.numel();
        int hiddenDim = 1;
        for (long d : normalizedShape)
            hiddenDim *= (int) d;

        HostAccessibleStorage outBuf = allocateOutput(n * 4);
        MemorySegment si = seg(input);
        MemorySegment so = outBuf.segment();
        MemorySegment sw = weight != null ? seg(weight) : null;
        MemorySegment sb = bias != null ? seg(bias) : null;

        long numRows = n / hiddenDim;
        for (long i = 0; i < numRows; i++) {
            long rowOff = i * hiddenDim;
            // Mean via SIMD reduceSum
            float sum = CpuOps.reduceSum(si.asSlice(rowOff * 4L, hiddenDim * 4L), hiddenDim);
            float mean = sum / hiddenDim;

            float sumSq = 0f;
            for (int j = 0; j < hiddenDim; j++) {
                float diff = si.get(ValueLayout.JAVA_FLOAT, (rowOff + j) * 4L) - mean;
                sumSq += diff * diff;
            }
            float invStd = (float) (1.0 / Math.sqrt(sumSq / hiddenDim + eps));

            for (int j = 0; j < hiddenDim; j++) {
                float val = si.get(ValueLayout.JAVA_FLOAT, (rowOff + j) * 4L);
                float norm = (val - mean) * invStd;
                if (sw != null)
                    norm *= sw.get(ValueLayout.JAVA_FLOAT, j * 4L);
                if (sb != null)
                    norm += sb.get(ValueLayout.JAVA_FLOAT, j * 4L);
                so.set(ValueLayout.JAVA_FLOAT, (rowOff + j) * 4L, norm);
            }
        }
        return out(shape, outBuf);
    }

    @Override
    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        Shape shape = input.shape();
        long n = shape.numel();
        int hiddenDim = (int) weight.shape().dim(0);
        HostAccessibleStorage outBuf = allocateOutput(n * 4);
        NormOps.rmsNorm(seg(input), seg(weight), outBuf.segment(), n, hiddenDim, eps);
        return out(shape, outBuf);
    }

    @Override
    public Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        throw new UnsupportedOperationException("CpuBackend RoPE not implemented yet");
    }

    @Override
    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias,
            Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        // Inference mode: normalize using running stats
        // input shape: [N, C, H, W] or [N, C]
        Shape shape = input.shape();
        long n = shape.numel();
        int channels = (int) shape.dim(1);
        long spatialSize = n / ((long) shape.dim(0) * channels);

        HostAccessibleStorage outBuf = allocateOutput(n * 4);
        MemorySegment si = seg(input);
        MemorySegment so = outBuf.segment();
        MemorySegment smean = seg(runningMean);
        MemorySegment svar = seg(runningVar);
        MemorySegment sw = weight != null ? seg(weight) : null;
        MemorySegment sb = bias != null ? seg(bias) : null;

        long numBatches = shape.dim(0);
        for (long b = 0; b < numBatches; b++) {
            for (int c = 0; c < channels; c++) {
                float mean = smean.get(ValueLayout.JAVA_FLOAT, c * 4L);
                float var = svar.get(ValueLayout.JAVA_FLOAT, c * 4L);
                float invStd = (float) (1.0 / Math.sqrt(var + eps));
                float gamma = sw != null ? sw.get(ValueLayout.JAVA_FLOAT, c * 4L) : 1f;
                float beta = sb != null ? sb.get(ValueLayout.JAVA_FLOAT, c * 4L) : 0f;

                long baseOff = (b * channels + c) * spatialSize;
                for (long s = 0; s < spatialSize; s++) {
                    long idx = (baseOff + s) * 4L;
                    float x = si.get(ValueLayout.JAVA_FLOAT, idx);
                    so.set(ValueLayout.JAVA_FLOAT, idx, gamma * (x - mean) * invStd + beta);
                }
            }
        }
        return out(shape, outBuf);
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    @Override
    public Tensor sum(Tensor a) {
        float total = CpuOps.reduceSum(seg(a), a.numel());
        HostAccessibleStorage o = allocateOutput(4);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, total);
        return out(new Shape(1), o);
    }

    @Override
    public Tensor sum(Tensor a, int dim, boolean keepDim) {
        return reduceAlongDim(a, dim, keepDim, true);
    }

    @Override
    public Tensor mean(Tensor a) {
        float total = CpuOps.reduceSum(seg(a), a.numel());
        HostAccessibleStorage o = allocateOutput(4);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, total / a.numel());
        return out(new Shape(1), o);
    }

    @Override
    public Tensor mean(Tensor a, int dim, boolean keepDim) {
        return reduceAlongDim(a, dim, keepDim, false);
    }

    @Override
    public Tensor max(Tensor a) {
        float best = CpuOps.reduceMax(seg(a), a.numel());
        HostAccessibleStorage o = allocateOutput(4);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, best);
        return out(new Shape(1), o);
    }

    /** Shared reduce-along-dim logic for sum and mean. */
    private Tensor reduceAlongDim(Tensor a, int dim, boolean keepDim, boolean isSum) {
        long[] inDims = a.shape().dims();
        int rank = inDims.length;
        if (dim < 0) {
            dim = rank + dim;
        }
        long dimLen = inDims[dim];

        long[] outDims = new long[keepDim ? rank : rank - 1];
        int outIdx = 0;
        for (int d = 0; d < rank; d++) {
            if (d == dim && !keepDim)
                continue;
            outDims[outIdx++] = (d == dim) ? 1 : inDims[d];
        }
        Shape outShape = new Shape(outDims);
        HostAccessibleStorage outBuf = allocateOutput(outShape.numel() * 4);
        MemorySegment src = seg(a);
        MemorySegment dst = outBuf.segment();

        // Compute input strides
        long[] strides = new long[rank];
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--)
            strides[d] = strides[d + 1] * inDims[d + 1];

        long outerSize = 1, innerSize = 1;
        for (int d = 0; d < dim; d++)
            outerSize *= inDims[d];
        for (int d = dim + 1; d < rank; d++)
            innerSize *= inDims[d];

        for (long o = 0; o < outerSize; o++) {
            for (long i = 0; i < innerSize; i++) {
                float acc = 0f;
                for (long k = 0; k < dimLen; k++) {
                    long inOff = (o * dimLen + k) * innerSize + i;
                    acc += src.get(ValueLayout.JAVA_FLOAT, inOff * 4L);
                }
                if (!isSum)
                    acc /= dimLen;
                dst.set(ValueLayout.JAVA_FLOAT, (o * innerSize + i) * 4L, acc);
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
        // pred: [N, C] log-probabilities (after log_softmax); target: [N] class indices
        // (I32)
        long batchSize = pred.shape().dim(0);
        long numClasses = pred.shape().dim(1);
        MemorySegment sp = seg(pred);
        MemorySegment st = seg(target);
        float loss = 0f;
        for (long i = 0; i < batchSize; i++) {
            int classIdx = st.get(ValueLayout.JAVA_INT, i * 4L);
            loss -= sp.get(ValueLayout.JAVA_FLOAT, (i * numClasses + classIdx) * 4L);
        }
        loss /= batchSize;
        HostAccessibleStorage o = allocateOutput(4);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, loss);
        return out(new Shape(1), o);
    }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
        // pred: [N] predicted probabilities; target: [N] binary labels in {0,1}
        long n = pred.numel();
        MemorySegment sp = seg(pred);
        MemorySegment st = seg(target);
        float loss = 0f;
        for (long i = 0; i < n; i++) {
            float p = Math.max(1e-7f, Math.min(1 - 1e-7f, CpuOps.getF(sp, i)));
            float t = CpuOps.getF(st, i);
            loss -= t * (float) Math.log(p) + (1 - t) * (float) Math.log(1 - p);
        }
        loss /= n;
        HostAccessibleStorage o = allocateOutput(4);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, loss);
        return out(new Shape(1), o);
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
                        computeAttentionGroup(q, k, v, mask, out, group,
                                queryLength, keyLength, headDim, scale, causal));
            } else {
                for (int group = 0; group < groupCount; group++) {
                    computeAttentionGroup(q, k, v, mask, out, group,
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
        for (long factor : new long[] {queryLength, keyLength, headDim}) {
            if (factor != 0 && work > Long.MAX_VALUE / factor) {
                return true;
            }
            work *= factor;
        }
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
            long queryLength,
            long keyLength,
            long headDim,
            float scale,
            boolean causal) {
        long qBase = (long) group * queryLength * headDim;
        long kBase = (long) group * keyLength * headDim;
        double[] weightedValues = new double[Math.toIntExact(headDim)];

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
                        q, (qBase + qi * headDim) * Float.BYTES,
                        k, (kBase + ki * headDim) * Float.BYTES,
                        Math.toIntExact(headDim)) * scale + additive;

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

                for (long d = 0; d < headDim; d++) {
                    float valueAtKey = v.get(ValueLayout.JAVA_FLOAT,
                            (kBase + ki * headDim + d) * Float.BYTES);
                    int index = Math.toIntExact(d);
                    weightedValues[index] = weightedValues[index] * previousScale
                            + currentScale * valueAtKey;
                }
                runningMax = nextMax;
            }

            for (long d = 0; d < headDim; d++) {
                float result = denominator == 0d
                        ? 0f : (float) (weightedValues[Math.toIntExact(d)] / denominator);
                out.set(ValueLayout.JAVA_FLOAT,
                        (qBase + qi * headDim + d) * Float.BYTES, result);
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
        if (input.shape().rank() != 4) {
            throw new IllegalArgumentException("upsampleNearest2d expects NCHW rank-4 input");
        }
        if (input.dtype() != DType.F32) {
            throw new UnsupportedOperationException(
                    "CPU nearest-neighbor upsampling currently supports F32 tensors only");
        }
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

        for (long n = 0; n < batch; n++) {
            for (long c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputH; oh++) {
                    long ih = Math.min(inputH - 1, (long) oh * inputH / outputH);
                    for (int ow = 0; ow < outputW; ow++) {
                        long iw = Math.min(inputW - 1, (long) ow * inputW / outputW);
                        long sourceIndex = ((n * channels + c) * inputH + ih) * inputW + iw;
                        long destinationIndex = ((n * channels + c) * outputH + oh) * outputW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, destinationIndex * Float.BYTES,
                                source.get(ValueLayout.JAVA_FLOAT, sourceIndex * Float.BYTES));
                    }
                }
            }
        }
        return out(outputShape, output);
    }

    @Override
    public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        long N = input.shape().dim(0);
        long C = input.shape().dim(1);
        long H = input.shape().dim(2);
        long W = input.shape().dim(3);
        long outH = (H + 2L * padding - kernelSize) / stride + 1;
        long outW = (W + 2L * padding - kernelSize) / stride + 1;

        Shape outShape = new Shape(N, C, outH, outW);
        HostAccessibleStorage outBuf = allocateOutput(outShape.numel() * 4);
        MemorySegment si = seg(input);
        MemorySegment so = outBuf.segment();

        for (long n = 0; n < N; n++) {
            for (long c = 0; c < C; c++) {
                for (long oh = 0; oh < outH; oh++) {
                    for (long ow = 0; ow < outW; ow++) {
                        float best = Float.NEGATIVE_INFINITY;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                long ih = oh * stride - padding + kh;
                                long iw = ow * stride - padding + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                    continue;
                                float v = si.get(ValueLayout.JAVA_FLOAT,
                                        ((n * C + c) * H + ih) * W * 4L + iw * 4L);
                                if (v > best)
                                    best = v;
                            }
                        }
                        so.set(ValueLayout.JAVA_FLOAT,
                                ((n * C + c) * outH + oh) * outW * 4L + ow * 4L, best);
                    }
                }
            }
        }
        return out(outShape, outBuf);
    }

    @Override
    public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        long N = input.shape().dim(0);
        long C = input.shape().dim(1);
        long H = input.shape().dim(2);
        long W = input.shape().dim(3);

        Shape outShape = new Shape(N, C, outputH, outputW);
        HostAccessibleStorage outBuf = allocateOutput(outShape.numel() * 4);
        MemorySegment si = seg(input);
        MemorySegment so = outBuf.segment();

        for (long n = 0; n < N; n++) {
            for (long c = 0; c < C; c++) {
                for (int oh = 0; oh < outputH; oh++) {
                    for (int ow = 0; ow < outputW; ow++) {
                        long hStart = (long) oh * H / outputH;
                        long hEnd = (long) (oh + 1) * H / outputH;
                        long wStart = (long) ow * W / outputW;
                        long wEnd = (long) (ow + 1) * W / outputW;
                        float acc = 0f;
                        long cnt = 0;
                        for (long ih = hStart; ih < hEnd; ih++) {
                            for (long iw = wStart; iw < wEnd; iw++) {
                                acc += si.get(ValueLayout.JAVA_FLOAT,
                                        ((n * C + c) * H + ih) * W * 4L + iw * 4L);
                                cnt++;
                            }
                        }
                        so.set(ValueLayout.JAVA_FLOAT,
                                ((n * C + c) * outputH + oh) * outputW * 4L + ow * 4L,
                                cnt > 0 ? acc / cnt : 0f);
                    }
                }
            }
        }
        return out(outShape, outBuf);
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
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("cat requires at least one tensor");
        }
        Tensor first = tensors.get(0);
        int rank = first.shape().rank();
        if (rank == 0) {
            throw new IllegalArgumentException("cat does not support scalar tensors");
        }
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("cat axis out of range: " + axis);
        }
        long[] outputDims = first.shape().dims();
        long axisTotal = 0;
        for (int i = 0; i < tensors.size(); i++) {
            Tensor tensor = tensors.get(i);
            if (tensor.dtype() != DType.F32) {
                throw new UnsupportedOperationException("CPU cat currently supports F32 tensors only");
            }
            if (tensor.shape().rank() != rank) {
                throw new IllegalArgumentException("cat rank mismatch at tensor " + i);
            }
            for (int d = 0; d < rank; d++) {
                if (d != axis && tensor.shape().dim(d) != first.shape().dim(d)) {
                    throw new IllegalArgumentException("cat shape mismatch at tensor " + i + " dimension " + d);
                }
            }
            axisTotal = Math.addExact(axisTotal, tensor.shape().dim(axis));
        }
        outputDims[axis] = axisTotal;
        long outer = 1;
        long inner = 1;
        for (int d = 0; d < axis; d++) outer = Math.multiplyExact(outer, first.shape().dim(d));
        for (int d = axis + 1; d < rank; d++) inner = Math.multiplyExact(inner, first.shape().dim(d));

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
                long blockElements = Math.multiplyExact(tensors.get(i).shape().dim(axis), inner);
                int blockLength = Math.toIntExact(blockElements);
                int sourceOffset = Math.toIntExact(Math.multiplyExact(outerIndex, blockElements));
                System.arraycopy(sources[i], sourceOffset, output, destination, blockLength);
                destination += blockLength;
            }
        }
        return TensorFactory.of(output, this, outputDims);
    }

}
