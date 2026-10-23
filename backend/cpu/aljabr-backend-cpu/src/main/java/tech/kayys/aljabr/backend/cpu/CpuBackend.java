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
import java.util.concurrent.ThreadLocalRandom;

/**
 * Pure-Java CPU backend for the Aljabr compute engine.
 *
 * <p>
 * All pointwise operations delegate to {@link CpuOps} which uses the JDK
 * Incubator Vector API ({@code jdk.incubator.vector}) for SIMD acceleration.
 * On ARM Apple-Silicon this maps to 128-bit NEON; on x86-64 with AVX-512 to
 * 512-bit lanes.
 *
 * <p>
 * An optional {@link OffHeapBufferPool} can be supplied to recycle
 * off-heap memory across inference steps and reduce GC pressure.
 */
public final class CpuBackend implements ComputeBackend {

    private final OffHeapBufferPool pool;

    public CpuBackend() {
        this(null);
    }

    public CpuBackend(OffHeapBufferPool pool) {
        this.pool = pool;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private CpuBuffer allocate(long byteSize) {
        if (pool != null) {
            return new CpuBuffer(pool.acquire(byteSize), pool.arena(), pool);
        }
        return new CpuBuffer(byteSize);
    }

    private static MemorySegment seg(Tensor t) {
        return ((DefaultTensor) t).buffer().segment();
    }

    private DefaultTensor out(Shape shape, CpuBuffer buf) {
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, buf, this);
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    @Override
    public Tensor add(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 0);
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.add(seg(a), seg(b), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 3);
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.sub(seg(a), seg(b), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.mulScalar(seg(a), scalar, o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        if (!a.shape().equals(b.shape()))
            return broadcastOp(a, b, 1);
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
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
        CpuBuffer o = allocate(n * 4);
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

        CpuBuffer o = allocate(numel * 4);
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
        CpuBuffer o = allocate(n * 4);
        CpuOps.addScalar(seg(a), scalar, o.segment(), n);
        return out(s, o);
    }

    // ── Matmul ────────────────────────────────────────────────────────────────

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        long[] shapeA = a.shape().dims();
        long[] shapeB = b.shape().dims();
        if (shapeA.length < 2 || shapeB.length < 2)
            throw new IllegalArgumentException("matmul requires at least 2D tensors");

        int m = (int) shapeA[shapeA.length - 2];
        int k = (int) shapeA[shapeA.length - 1];
        int k2 = (int) shapeB[shapeB.length - 2];
        int n = (int) shapeB[shapeB.length - 1];
        if (k != k2)
            throw new IllegalArgumentException("Dimension mismatch: " + k + " != " + k2);

        long[] outDims = shapeA.clone();
        outDims[outDims.length - 2] = m;
        outDims[outDims.length - 1] = n;
        Shape outShape = new Shape(outDims);
        CpuBuffer outBuf = allocate(outShape.numel() * 4);

        // Zero-init output
        outBuf.segment().fill((byte) 0);

        MatmulCpu.matmul(seg(a), seg(b), outBuf.segment(), m, n, k);
        return out(outShape, outBuf);
    }

    // ── Shape ops ────────────────────────────────────────────────────────────

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        // Zero-copy: retain the shared buffer until every tensor view is released.
        Buffer buffer = ((DefaultTensor) a).buffer();
        buffer.retain();
        return new DefaultTensor(new Shape(newShape), a.dtype(), a.device(), buffer, this);
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        // Attempt zero-copy view when the slice is contiguous in row-major layout.
        Shape inShape = a.shape();
        int rank = inShape.rank();
        long[] inDims = inShape.dims();

        Shape outShape = new Shape(sizes);

        // Strides in elements (row-major) for the input
        long[] baseStrides = new long[rank];
        baseStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--)
            baseStrides[i] = baseStrides[i + 1] * inDims[i + 1];

        // Expected contiguous strides for the view
        long[] viewExpected = new long[rank];
        long exp = 1;
        for (int i = rank - 1; i >= 0; i--) {
            viewExpected[i] = exp;
            exp *= sizes[i];
        }

        boolean isContiguous = true;
        for (int i = 0; i < rank; i++) {
            if (baseStrides[i] != viewExpected[i]) {
                isContiguous = false;
                break;
            }
        }

        MemorySegment src = seg(a);
        long elemOffset = 0;
        for (int i = 0; i < rank; i++)
            elemOffset += offsets[i] * baseStrides[i];
        long byteOffset = elemOffset * 4L;
        long byteSize = outShape.numel() * 4L;

        if (isContiguous) {
            // Create zero-copy view into the existing segment
            MemorySegment viewSeg = src.asSlice(byteOffset, byteSize);
            Buffer parentBuffer = ((DefaultTensor) a).buffer();
            CpuBuffer viewBuf = new CpuBuffer(viewSeg, parentBuffer);
            return new DefaultTensor(outShape, a.dtype(), a.device(), viewBuf, this);
        }

        // Fallback: copy the selected region into a new buffer (existing behavior)
        CpuBuffer outBuf = allocate(outShape.numel() * 4);

        long outIdx = 0;
        long totalOut = outShape.numel();

        long[] cursor = new long[rank];
        for (long elem = 0; elem < totalOut; elem++) {
            long inIdx = 0;
            for (int d = 0; d < rank; d++)
                inIdx += (offsets[d] + cursor[d]) * baseStrides[d];
            outBuf.segment().set(ValueLayout.JAVA_FLOAT, outIdx * 4L,
                    src.get(ValueLayout.JAVA_FLOAT, inIdx * 4L));
            outIdx++;
            for (int d = rank - 1; d >= 0; d--) {
                if (++cursor[d] < sizes[d])
                    break;
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
        CpuBuffer outBuf = allocate(outShape.numel() * 4);

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
        CpuBuffer o = allocate(n * 4);
        CpuOps.relu(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor sigmoid(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.sigmoid(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor tanh(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.tanh(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor log(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.log(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor exp(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.exp(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor silu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        CpuOps.silu(seg(a), o.segment(), n);
        return out(s, o);
    }

    @Override
    public Tensor gelu(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
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
        CpuBuffer outBuf = allocate(n * 4);
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

        CpuBuffer outBuf = allocate(n * 4);
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
        CpuBuffer outBuf = allocate(n * 4);
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

        CpuBuffer outBuf = allocate(n * 4);
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
        CpuBuffer o = allocate(4);
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
        CpuBuffer o = allocate(4);
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
        CpuBuffer o = allocate(4);
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
        CpuBuffer outBuf = allocate(outShape.numel() * 4);
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
        CpuBuffer o = allocate(n * 4);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++)
            CpuOps.setF(dst, i, (float) Math.pow(CpuOps.getF(src, i), exponent));
        return out(s, o);
    }

    @Override
    public Tensor abs(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++)
            CpuOps.setF(dst, i, Math.abs(CpuOps.getF(src, i)));
        return out(s, o);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        Shape s = a.shape();
        long n = s.numel();
        CpuBuffer o = allocate(n * 4);
        MemorySegment src = seg(a), dst = o.segment();
        for (long i = 0; i < n; i++)
            CpuOps.setF(dst, i, (float) Math.sqrt(CpuOps.getF(src, i)));
        return out(s, o);
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        Shape s = a.shape();
        CpuBuffer o = allocate(s.numel() * 4);
        o.segment().fill((byte) 0);
        return out(s, o);
    }

    @Override
    public Tensor cast(Tensor a, DType dtype) {
        if (a.dtype() == dtype)
            return a;

        long n = a.numel();
        CpuBuffer outBuf = allocate(dtype.memoryFootprintBytes(n));

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
        CpuBuffer o = allocate(n * 4);
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
        CpuBuffer o = allocate(4);
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
        CpuBuffer o = allocate(4);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, loss);
        return out(new Shape(1), o);
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        try {
            return FlashAttentionCpu.forward(Q, K, V, Runtime.getRuntime().availableProcessors());
        } catch (Exception e) {
            // Fallback: naive O(n²) attention
            Tensor scores = matmul(Q, transpose(K));
            float scale = (float) (1.0 / Math.sqrt(Q.shape().dim(Q.shape().rank() - 1)));
            scores = mul(scores, scale);
            Tensor probs = softmax(scores);
            return matmul(probs, V);
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
        CpuBuffer outBuf = allocate(outShape.numel() * 4);
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
    public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        long N = input.shape().dim(0);
        long C = input.shape().dim(1);
        long H = input.shape().dim(2);
        long W = input.shape().dim(3);
        long outH = (H + 2L * padding - kernelSize) / stride + 1;
        long outW = (W + 2L * padding - kernelSize) / stride + 1;

        Shape outShape = new Shape(N, C, outH, outW);
        CpuBuffer outBuf = allocate(outShape.numel() * 4);
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
        CpuBuffer outBuf = allocate(outShape.numel() * 4);
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
        CpuBuffer outBuf = allocate(numTokens * embeddingDim * 4);
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
}