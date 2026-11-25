package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.backend.cpu.ops.CpuOps;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.LongFunction;

import tech.kayys.aljabr.backend.cpu.ops.CpuOps;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/**
 * CPU reduction and probability-normalization execution.
 *
 * <p>Owns reduction-axis semantics and stable softmax/log-softmax kernels.
 * The backend facade supplies storage allocation and tensor construction.</p>
 */
final class CpuReductionExecution {
    @FunctionalInterface
    interface OutputFactory {
        Tensor create(Shape shape, HostAccessibleStorage storage);
    }

    private final LongFunction<HostAccessibleStorage> outputAllocator;
    private final OutputFactory outputFactory;

    CpuReductionExecution(
            LongFunction<HostAccessibleStorage> outputAllocator,
            OutputFactory outputFactory) {
        this.outputAllocator = Objects.requireNonNull(outputAllocator, "outputAllocator");
        this.outputFactory = Objects.requireNonNull(outputFactory, "outputFactory");
    }

    public Tensor sum(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        CpuTensorLayoutAccess.requireF32(a);
        MemorySegment source = CpuTensorStorageAccess.segment(a);
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
        HostAccessibleStorage o = outputAllocator.apply(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, total);
        return outputFactory.create(new Shape(1), o);
    }

    public Tensor sum(Tensor a, int dim, boolean keepDim) {
        return reduceAlongDim(a, dim, keepDim, true);
    }

    public Tensor mean(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        if (a.numel() == 0) {
            throw new IllegalArgumentException("Mean is undefined for an empty tensor");
        }
        CpuTensorLayoutAccess.requireF32(a);
        MemorySegment source = CpuTensorStorageAccess.segment(a);
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
        HostAccessibleStorage o = outputAllocator.apply(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, total / a.numel());
        return outputFactory.create(new Shape(1), o);
    }

    public Tensor mean(Tensor a, int dim, boolean keepDim) {
        return reduceAlongDim(a, dim, keepDim, false);
    }

    public Tensor max(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        CpuTensorLayoutAccess.requireF32(a);
        MemorySegment source = CpuTensorStorageAccess.segment(a);
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
        HostAccessibleStorage o = outputAllocator.apply(Float.BYTES);
        o.segment().set(ValueLayout.JAVA_FLOAT, 0, best);
        return outputFactory.create(new Shape(1), o);
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
        HostAccessibleStorage outBuf = outputAllocator.apply(
                Math.multiplyExact(outShape.numel(), (long) Float.BYTES));
        MemorySegment source = CpuTensorStorageAccess.segment(a);
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
        return outputFactory.create(outShape, outBuf);
    }

    public Tensor softmax(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        return softmax(a, a.shape().rank() - 1);
    }

    public Tensor softmax(Tensor a, int dim) {
        return normalizeAlongSuffix(a, dim, false);
    }

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
        HostAccessibleStorage output = outputAllocator.apply(
                Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) {
            return outputFactory.create(shape, output);
        }

        MemorySegment source = CpuTensorStorageAccess.segment(a);
        MemorySegment destination = output.segment();
        TensorLayout layout = a.layout();
        if (!logarithmic && layout.isContiguous() && layout.storageOffset() == 0
                && rowLength <= Integer.MAX_VALUE) {
            int contiguousRowLength = Math.toIntExact(rowLength);
            for (long row = 0; row < rowCount; row++) {
                CpuOps.softmax(source, destination, Math.multiplyExact(row, rowLength),
                        contiguousRowLength);
            }
            return outputFactory.create(shape, output);
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
        return outputFactory.create(shape, output);
    }
}
