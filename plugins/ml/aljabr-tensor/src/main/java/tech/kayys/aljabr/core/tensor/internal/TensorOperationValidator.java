package tech.kayys.aljabr.core.tensor.internal;

import java.util.List;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.diagnostics.Diagnostic;
import tech.kayys.aljabr.diagnostics.DiagnosticSeverity;
import tech.kayys.aljabr.diagnostics.DiagnosticSink;

/** Shared structural validation for concatenation at tensor and backend boundaries. */
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
            throw new IllegalArgumentException("Dimension mismatch: " + k + " != " + k2);
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
                throw new IllegalArgumentException(
                        "matmul batch dimensions are not broadcast-compatible at aligned axis "
                                + axis + ": " + dimA + " != " + dimB);
            }
            batchDims[axis] = Math.max(dimA, dimB);
        }
        long[] outputDims = java.util.Arrays.copyOf(batchDims, batchRank + 2);
        outputDims[batchRank] = m;
        outputDims[batchRank + 1] = n;
        return new MatmulPlan(batchDims, outputDims, m, k, n);
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
