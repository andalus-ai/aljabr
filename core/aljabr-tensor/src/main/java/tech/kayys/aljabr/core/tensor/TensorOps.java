package tech.kayys.aljabr.core.tensor;

import java.util.List;
import java.util.Objects;

/**
 * Tensor operations that are not tied to a single tensor instance.
 *
 * <p>Operations in this class must either return a mathematically valid result
 * or fail explicitly when the requested operation is not implemented.
 */
public final class TensorOps {
    private TensorOps() {
    }

    /**
     * Einstein summation is not implemented yet. In particular, this method must
     * not return an input tensor as a placeholder because that silently produces
     * a result unrelated to the requested equation.
     *
     * @throws UnsupportedOperationException until einsum contraction support lands
     */
    public static Tensor einsum(String equation, Tensor... tensors) {
        Objects.requireNonNull(equation, "equation must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        if (equation.isBlank()) {
            throw new IllegalArgumentException("equation must not be blank");
        }
        if (tensors.length == 0) {
            throw new IllegalArgumentException("einsum requires at least one tensor");
        }
        for (int i = 0; i < tensors.length; i++) {
            Objects.requireNonNull(tensors[i], "tensor at index " + i + " must not be null");
        }
        throw new UnsupportedOperationException(
                "TensorOps.einsum is not implemented yet; refusing to return a misleading placeholder result");
    }

    /**
     * Concatenates F32 CPU tensors along an existing dimension.
     *
     * <p>All inputs must have equal rank and equal dimensions except for the
     * concatenation axis. This first implementation deliberately supports only
     * F32 CPU tensors; other dtypes/devices fail explicitly rather than silently
     * copying or converting data.
     *
     * @param tensors non-empty ordered list of tensors to concatenate
     * @param dim axis along which to concatenate; negative axes count from the end
     * @return a new concatenated tensor
     */
    public static Tensor cat(List<Tensor> tensors, int dim) {
        Objects.requireNonNull(tensors, "tensors must not be null");
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("cat requires at least one tensor");
        }

        Tensor first = Objects.requireNonNull(tensors.get(0), "tensor at index 0 must not be null");
        int rank = first.shape().rank();
        if (rank == 0) {
            throw new IllegalArgumentException("cat does not support scalar tensors");
        }
        int axis = dim < 0 ? rank + dim : dim;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("dimension " + dim + " out of range for rank " + rank);
        }
        if (first.dtype() != DType.F32 || first.device() != DeviceType.CPU) {
            throw new UnsupportedOperationException("cat currently supports F32 CPU tensors only");
        }

        long[] outputDims = first.shape().dims();
        long axisTotal = 0;
        for (int i = 0; i < tensors.size(); i++) {
            Tensor tensor = Objects.requireNonNull(tensors.get(i), "tensor at index " + i + " must not be null");
            if (tensor.shape().rank() != rank) {
                throw new IllegalArgumentException("rank mismatch at tensor " + i + ": expected " + rank
                        + " but got " + tensor.shape().rank());
            }
            if (tensor.dtype() != first.dtype() || tensor.device() != first.device()) {
                throw new IllegalArgumentException("dtype/device mismatch at tensor " + i);
            }
            for (int d = 0; d < rank; d++) {
                if (d != axis && tensor.shape().dim(d) != first.shape().dim(d)) {
                    throw new IllegalArgumentException("shape mismatch at tensor " + i + " dimension " + d
                            + ": expected " + first.shape().dim(d) + " but got " + tensor.shape().dim(d));
                }
            }
            try {
                axisTotal = Math.addExact(axisTotal, tensor.shape().dim(axis));
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("concatenated dimension overflows long", e);
            }
        }
        outputDims[axis] = axisTotal;
        long outputElements = new Shape(outputDims).numel();
        if (outputElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("cat result is too large for a Java float[]: " + outputElements);
        }

        long outer = 1;
        long inner = 1;
        for (int d = 0; d < axis; d++) {
            outer = Math.multiplyExact(outer, first.shape().dim(d));
        }
        for (int d = axis + 1; d < rank; d++) {
            inner = Math.multiplyExact(inner, first.shape().dim(d));
        }

        float[] output = new float[Math.toIntExact(outputElements)];
        int destination = 0;
        for (long outerIndex = 0; outerIndex < outer; outerIndex++) {
            for (Tensor tensor : tensors) {
                long blockElements = Math.multiplyExact(tensor.shape().dim(axis), inner);
                float[] source = tensor.toFloatArray();
                long sourceOffset = Math.multiplyExact(outerIndex, blockElements);
                int blockLength = Math.toIntExact(blockElements);
                System.arraycopy(source, Math.toIntExact(sourceOffset), output, destination, blockLength);
                destination += blockLength;
            }
        }
        return Tensor.of(output, outputDims);
    }
}
