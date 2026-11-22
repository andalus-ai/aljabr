package tech.kayys.aljabr.core.tensor.internal;

import java.util.List;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.diagnostics.Diagnostic;
import tech.kayys.aljabr.diagnostics.DiagnosticSeverity;
import tech.kayys.aljabr.diagnostics.DiagnosticSink;

/** Shared structural validation for tensor operations at tensor and backend boundaries. */
public final class TensorOperationValidator {
    private TensorOperationValidator() { }

    public static CatPlan validateCat(List<Tensor> tensors, int requestedAxis, DiagnosticSink sink) {
        Objects.requireNonNull(sink, "diagnosticSink must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        if (tensors.isEmpty()) throw new IllegalArgumentException("cat requires at least one tensor");
        Tensor first = Objects.requireNonNull(tensors.get(0), "tensor at index 0 must not be null");
        int rank = first.shape().rank();
        if (rank == 0) throw new IllegalArgumentException("cat does not support scalar tensors");
        int axis = requestedAxis < 0 ? rank + requestedAxis : requestedAxis;
        if (axis < 0 || axis >= rank) throw new IllegalArgumentException("cat axis out of range: " + requestedAxis);

        long[] outputDims = first.shape().dims();
        long axisTotal = 0;
        for (int i = 0; i < tensors.size(); i++) {
            Tensor tensor = Objects.requireNonNull(tensors.get(i), "tensor at index " + i + " must not be null");
            if (tensor.shape().rank() != rank) {
                throw failure(sink, "tensor.operation.rank-mismatch",
                        "rank mismatch at tensor " + i + ": expected " + rank + " but got " + tensor.shape().rank(),
                        Map.of("operation", "cat", "operandIndex", Integer.toString(i),
                                "expectedRank", Integer.toString(rank),
                                "actualRank", Integer.toString(tensor.shape().rank())));
            }
            if (tensor.dtype() != first.dtype() || tensor.device() != first.device()) {
                throw failure(sink, "tensor.operation.dtype-device-mismatch", "dtype/device mismatch at tensor " + i,
                        Map.of("operation", "cat", "operandIndex", Integer.toString(i),
                                "expectedDType", first.dtype().name(), "actualDType", tensor.dtype().name(),
                                "expectedDevice", first.device().name(), "actualDevice", tensor.device().name()));
            }
            if (tensor.backend() != first.backend()) {
                throw failure(sink, "tensor.operation.backend-mismatch", "compute backend mismatch at tensor " + i,
                        Map.of("operation", "cat", "operandIndex", Integer.toString(i),
                                "backendType", tensor.backend().getClass().getName()));
            }
            for (int d = 0; d < rank; d++) {
                if (d != axis && tensor.shape().dim(d) != first.shape().dim(d)) {
                    throw failure(sink, "tensor.operation.shape-mismatch",
                            "shape mismatch at tensor " + i + " dimension " + d
                                    + ": expected " + first.shape().dim(d) + " but got " + tensor.shape().dim(d),
                            Map.of("operation", "cat", "operandIndex", Integer.toString(i),
                                    "axis", Integer.toString(d),
                                    "expectedDimension", Long.toString(first.shape().dim(d)),
                                    "actualDimension", Long.toString(tensor.shape().dim(d))));
                }
            }
            try {
                axisTotal = Math.addExact(axisTotal, tensor.shape().dim(axis));
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException("concatenated dimension overflows long", overflow);
            }
        }
        outputDims[axis] = axisTotal;
        return new CatPlan(first, axis, outputDims);
    }

    /**
     * Validates the structural shape contract for matrix multiplication and
     * returns the broadcasted batch and output dimensions. Dtype and kernel
     * capability checks deliberately remain backend responsibilities.
     */
    public static MatmulPlan validateMatmul(Shape left, Shape right) {
        return validateMatmul(left, right, DiagnosticSink.noop());
    }

    /** Reports matrix-dimension and batch-broadcast contract failures to the supplied sink. */
    public static MatmulPlan validateMatmul(Shape left, Shape right, DiagnosticSink sink) {
        Objects.requireNonNull(sink, "diagnosticSink must not be null");
        Objects.requireNonNull(left, "left shape must not be null");
        Objects.requireNonNull(right, "right shape must not be null");
        long[] a = left.dims();
        long[] b = right.dims();
        if (a.length < 2 || b.length < 2) {
            throw new IllegalArgumentException("matmul requires at least 2D tensors");
        }
        long m = a[a.length - 2];
        long k = a[a.length - 1];
        long k2 = b[b.length - 2];
        long n = b[b.length - 1];
        if (k != k2) {
            throw failure(sink, "tensor.operation.dimension-mismatch",
                    "Dimension mismatch: " + k + " != " + k2,
                    Map.of("operation", "matmul", "axis", "contracted",
                            "expectedDimension", Long.toString(k),
                            "actualDimension", Long.toString(k2)));
        }

        int batchRankA = a.length - 2;
        int batchRankB = b.length - 2;
        int batchRank = Math.max(batchRankA, batchRankB);
        long[] batchDims = new long[batchRank];
        for (int axis = 0; axis < batchRank; axis++) {
            int axisA = axis - (batchRank - batchRankA);
            int axisB = axis - (batchRank - batchRankB);
            long dimA = axisA < 0 ? 1 : a[axisA];
            long dimB = axisB < 0 ? 1 : b[axisB];
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw failure(sink, "tensor.operation.broadcast-mismatch",
                        "matmul batch dimensions are not broadcast-compatible at aligned axis "
                                + axis + ": " + dimA + " != " + dimB,
                        Map.of("operation", "matmul", "axis", Integer.toString(axis),
                                "leftDimension", Long.toString(dimA),
                                "rightDimension", Long.toString(dimB)));
            }
            batchDims[axis] = Math.max(dimA, dimB);
        }
        long[] outputDims = java.util.Arrays.copyOf(batchDims, batchRank + 2);
        outputDims[batchRank] = m;
        outputDims[batchRank + 1] = n;
        return new MatmulPlan(batchDims, outputDims, m, k, n);
    }

    /**
     * Validates NumPy-style trailing-axis broadcasting for elementwise binary
     * operations. Kernel and dtype support remain backend responsibilities.
     */
    public static BroadcastPlan validateBroadcast(Shape left, Shape right) {
        return validateBroadcast(left, right, DiagnosticSink.noop());
    }

    /** Reports incompatible dimensions to the supplied sink. */
    public static BroadcastPlan validateBroadcast(Shape left, Shape right, DiagnosticSink sink) {
        Objects.requireNonNull(sink, "diagnosticSink must not be null");
        Objects.requireNonNull(left, "left shape must not be null");
        Objects.requireNonNull(right, "right shape must not be null");
        long[] a = left.dims();
        long[] b = right.dims();
        int rank = Math.max(a.length, b.length);
        long[] output = new long[rank];
        long elements = 1;
        for (int axis = 0; axis < rank; axis++) {
            int aIndex = axis - (rank - a.length);
            int bIndex = axis - (rank - b.length);
            long dimA = aIndex < 0 ? 1 : a[aIndex];
            long dimB = bIndex < 0 ? 1 : b[bIndex];
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw failure(sink, "tensor.operation.broadcast-mismatch",
                        "shapes are not broadcast-compatible at axis " + axis + ": " + dimA + " != " + dimB,
                        Map.of("operation", "broadcast", "axis", Integer.toString(axis),
                                "leftDimension", Long.toString(dimA),
                                "rightDimension", Long.toString(dimB)));
            }
            // A singleton expands to the other dimension, including zero.
            output[axis] = dimA == 1 ? dimB : dimA;
            try {
                elements = Math.multiplyExact(elements, output[axis]);
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException("broadcast output element count overflows long", overflow);
            }
        }
        return new BroadcastPlan(output, elements);
    }


    /**
     * Validates a reshape request without imposing backend-specific layout rules.
     * Reshape dimensions must be explicit and non-negative; inferred dimensions
     * are not part of the current tensor API contract.
     */
    public static Shape validateReshape(Shape input, long... requestedDimensions) {
        Objects.requireNonNull(input, "input shape must not be null");
        Shape output = new Shape(requestedDimensions);
        if (output.numel() != input.numel()) {
            throw new IllegalArgumentException(
                    "reshape must preserve the number of elements: input="
                            + input.numel() + ", output=" + output.numel());
        }
        return output;
    }


    /**
     * Validates an equal-sized split and returns the normalized axis, chunk size,
     * and logical shape of each output. Backends retain responsibility for views,
     * strides, storage ownership, and device-specific execution.
     */
    public static SplitPlan validateSplit(Shape input, int requestedAxis, int parts) {
        Objects.requireNonNull(input, "input shape must not be null");
        int rank = input.rank();
        long normalized = requestedAxis < 0 ? (long) rank + requestedAxis : requestedAxis;
        if (rank == 0 || normalized < 0 || normalized >= rank) {
            throw new IllegalArgumentException(
                    "Split axis " + requestedAxis + " is out of range for tensor rank " + rank);
        }
        if (parts <= 0) {
            throw new IllegalArgumentException("split parts must be positive; got " + parts);
        }
        int axis = (int) normalized;
        long axisSize = input.dim(axis);
        if (axisSize <= 0 || parts > axisSize || axisSize % parts != 0) {
            throw new IllegalArgumentException(
                    "split requires a positive axis size evenly divisible by parts; axis size="
                            + axisSize + ", parts=" + parts);
        }
        long chunkSize = axisSize / parts;
        long[] outputDimensions = input.dims();
        outputDimensions[axis] = chunkSize;
        return new SplitPlan(axis, parts, chunkSize, new Shape(outputDimensions));
    }

    public record SplitPlan(int axis, int parts, long chunkSize, Shape outputShape) { }

    /**
     * Validates axis-wise index selection. The index tensor must be rank one;
     * value bounds and index dtype are checked by the backend that can access
     * the index representation.
     */
    public static IndexSelectPlan validateIndexSelect(Shape input, Shape indices, int requestedAxis) {
        Objects.requireNonNull(input, "input shape must not be null");
        Objects.requireNonNull(indices, "indices shape must not be null");
        int rank = input.rank();
        long normalized = requestedAxis < 0 ? (long) rank + requestedAxis : requestedAxis;
        if (rank == 0 || normalized < 0 || normalized >= rank) {
            throw new IllegalArgumentException(
                    "Index-select axis " + requestedAxis + " is out of range for tensor rank " + rank);
        }
        if (indices.rank() != 1) {
            throw new IllegalArgumentException(
                    "index-select indices must be rank one; got rank " + indices.rank());
        }
        int axis = (int) normalized;
        long[] outputDimensions = input.dims();
        outputDimensions[axis] = indices.dim(0);
        return new IndexSelectPlan(axis, new Shape(outputDimensions));
    }

    public record IndexSelectPlan(int axis, Shape outputShape) { }

    /** Validates indexed addition metadata; index value bounds are backend-validated. */
    public static IndexAddPlan validateIndexAdd(Shape destination, Shape indices, Shape source, int requestedAxis) {
        Objects.requireNonNull(destination, "destination shape must not be null");
        Objects.requireNonNull(indices, "indices shape must not be null");
        Objects.requireNonNull(source, "source shape must not be null");
        int rank = destination.rank();
        long normalized = requestedAxis < 0 ? (long) rank + requestedAxis : requestedAxis;
        if (rank == 0 || normalized < 0 || normalized >= rank) {
            throw new IllegalArgumentException(
                    "Index-add axis " + requestedAxis + " is out of range for tensor rank " + rank);
        }
        if (indices.rank() != 1) {
            throw new IllegalArgumentException(
                    "index-add indices must be rank one; got rank " + indices.rank());
        }
        if (source.rank() != rank) {
            throw new IllegalArgumentException(
                    "index-add source rank " + source.rank() + " does not match destination rank " + rank);
        }
        int axis = (int) normalized;
        for (int dimension = 0; dimension < rank; dimension++) {
            long expected = dimension == axis ? indices.dim(0) : destination.dim(dimension);
            if (source.dim(dimension) != expected) {
                throw new IllegalArgumentException("index-add source dimension " + dimension + " is "
                        + source.dim(dimension) + "; expected " + expected);
            }
        }
        return new IndexAddPlan(axis, destination);
    }

    public record IndexAddPlan(int axis, Shape outputShape) { }

    /**
     * Validates flattening the complete logical tensor into a one-dimensional shape.
     * Scalars become a length-one vector and empty tensors remain empty vectors.
     */
    public static Shape validateFlatten(Shape input) {
        Objects.requireNonNull(input, "input shape must not be null");
        return new Shape(input.numel());
    }

    /**
     * Validates the default transpose operation, which swaps the final two axes.
     */
    public static TransposePlan validateTranspose(Shape input) {
        Objects.requireNonNull(input, "input shape must not be null");
        int rank = input.rank();
        if (rank < 2) {
            throw new IllegalArgumentException(
                    "transpose requires a tensor with rank >= 2; got " + rank);
        }
        return validateTranspose(input, rank - 2, rank - 1);
    }

    /**
     * Validates two transpose axes and returns their normalized positions and output shape.
     * Negative axes count from the final dimension; equal axes are a valid no-op.
     */
    public static TransposePlan validateTranspose(Shape input, int requestedAxis0, int requestedAxis1) {
        Objects.requireNonNull(input, "input shape must not be null");
        int rank = input.rank();
        long axis0 = requestedAxis0 < 0 ? (long) rank + requestedAxis0 : requestedAxis0;
        long axis1 = requestedAxis1 < 0 ? (long) rank + requestedAxis1 : requestedAxis1;
        if (rank == 0 || axis0 < 0 || axis0 >= rank || axis1 < 0 || axis1 >= rank) {
            throw new IllegalArgumentException(
                    "transpose dimensions must be in [-" + rank + ", " + rank + "); got d0="
                            + requestedAxis0 + ", d1=" + requestedAxis1);
        }
        int normalizedAxis0 = (int) axis0;
        int normalizedAxis1 = (int) axis1;
        long[] outputDimensions = input.dims();
        long temporary = outputDimensions[normalizedAxis0];
        outputDimensions[normalizedAxis0] = outputDimensions[normalizedAxis1];
        outputDimensions[normalizedAxis1] = temporary;
        return new TransposePlan(normalizedAxis0, normalizedAxis1, new Shape(outputDimensions));
    }

    public record TransposePlan(int axis0, int axis1, Shape outputShape) { }
    /**
     * Validates an unsqueeze axis and returns its normalized position and output shape.
     * Negative axes are counted from the end of the rank-plus-one output dimensions.
     */
    public static UnsqueezePlan validateUnsqueeze(Shape input, int requestedAxis) {
        Objects.requireNonNull(input, "input shape must not be null");
        int rank = input.rank();
        long normalized = requestedAxis < 0 ? (long) rank + 1 + requestedAxis : requestedAxis;
        if (normalized < 0 || normalized > rank) {
            throw new IllegalArgumentException(
                    "unsqueeze axis " + requestedAxis + " is out of range for tensor rank " + rank);
        }
        int axis = (int) normalized;
        long[] outputDimensions = new long[rank + 1];
        long[] inputDimensions = input.dims();
        System.arraycopy(inputDimensions, 0, outputDimensions, 0, axis);
        outputDimensions[axis] = 1;
        System.arraycopy(inputDimensions, axis, outputDimensions, axis + 1, rank - axis);
        return new UnsqueezePlan(axis, new Shape(outputDimensions));
    }

    /**
     * Returns the shape produced by removing every singleton dimension.
     * The legacy tensor contract represents an all-singleton (including scalar)
     * result as a one-element vector rather than a rank-zero scalar.
     */
    public static Shape validateSqueeze(Shape input) {
        Objects.requireNonNull(input, "input shape must not be null");
        long[] outputDimensions = java.util.Arrays.stream(input.dims())
                .filter(dimension -> dimension != 1)
                .toArray();
        return new Shape(outputDimensions.length == 0 ? new long[] { 1 } : outputDimensions);
    }

    public record UnsqueezePlan(int axis, Shape outputShape) { }

    /**
     * Validates slice coordinates and returns the logical output shape.
     * Storage offsets, strides, device constraints, and ownership remain backend concerns.
     */
    public static Shape validateSlice(Shape input, long[] offsets, long[] sizes) {
        Objects.requireNonNull(input, "input shape must not be null");
        int rank = input.rank();
        if (rank == 0) {
            throw new IllegalArgumentException("Cannot slice a scalar tensor");
        }
        if (offsets == null || sizes == null || offsets.length != rank || sizes.length != rank) {
            throw new IllegalArgumentException(
                    "Slice offsets and sizes must both have one entry per tensor dimension");
        }

        Shape output = new Shape(sizes);
        for (int axis = 0; axis < rank; axis++) {
            long offset = offsets[axis];
            long size = sizes[axis];
            long dimension = input.dim(axis);
            if (offset < 0 || offset > dimension || size > dimension - offset) {
                throw new IllegalArgumentException(
                        "Slice is outside tensor bounds at dimension " + axis
                                + ": offset=" + offset + ", size=" + size
                                + ", dimension=" + dimension);
            }
        }
        return output;
    }

    /**
     * Validates a single-axis reduction and returns its normalized axis and output shape.
     * Mean reductions reject empty axes; sum reductions preserve the additive identity
     * semantics for empty axes.
     */
    public static ReductionPlan validateReduction(Shape input, int requestedAxis, boolean mean) {
        Objects.requireNonNull(input, "input shape must not be null");
        int rank = input.rank();
        int axis = requestedAxis < 0 ? rank + requestedAxis : requestedAxis;
        if (rank == 0 || axis < 0 || axis >= rank) {
            throw new IllegalArgumentException(
                    "Reduction axis " + requestedAxis + " is out of range for tensor rank " + rank);
        }
        long reductionSize = input.dim(axis);
        if (mean && reductionSize == 0) {
            throw new IllegalArgumentException("Mean is undefined for an empty reduction dimension");
        }
        return new ReductionPlan(axis, reductionSize);
    }

    public record ReductionPlan(int axis, long reductionSize) { }

    public record BroadcastPlan(long[] outputDimensions, long outputElements) {
        public BroadcastPlan {
            outputDimensions = outputDimensions.clone();
        }
        @Override public long[] outputDimensions() { return outputDimensions.clone(); }
    }

    public record MatmulPlan(long[] batchDimensions, long[] outputDimensions, long rows, long inner, long columns) {
        public MatmulPlan {
            batchDimensions = batchDimensions.clone();
            outputDimensions = outputDimensions.clone();
        }
        @Override public long[] batchDimensions() { return batchDimensions.clone(); }
        @Override public long[] outputDimensions() { return outputDimensions.clone(); }
    }

    private static IllegalArgumentException failure(
            DiagnosticSink sink, String code, String message, Map<String, String> context) {
        sink.accept(new Diagnostic(code, DiagnosticSeverity.ERROR, "tensor.operation.validation", message, context));
        return new IllegalArgumentException(message);
    }

    public record CatPlan(Tensor first, int axis, long[] outputDimensions) {
        public CatPlan { outputDimensions = outputDimensions.clone(); }
        @Override public long[] outputDimensions() { return outputDimensions.clone(); }
    }
}
