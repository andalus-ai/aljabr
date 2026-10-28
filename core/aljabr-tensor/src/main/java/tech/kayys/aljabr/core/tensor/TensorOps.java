package tech.kayys.aljabr.core.tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Tensor operations that are not tied to a single tensor instance.
 *
 * <p>Operations in this class must either return a mathematically valid result
 * or fail explicitly when the requested operation is not implemented.
 */
public final class TensorOps {
    private static final int LABEL_COUNT = 52;

    private TensorOps() {
    }

    /**
     * Computes Einstein summation for F32 CPU tensors using an explicit equation.
     *
     * <p>The reference implementation supports single-letter ASCII labels,
     * explicit output labels, arbitrary operand counts, transposes, reductions,
     * outer products, and contractions. A label may occur at most once in each
     * operand and once in the output. Ellipses, implicit output labels, repeated
     * labels (diagonals), non-F32 tensors, and non-CPU tensors are rejected.
     *
     * <p>This implementation prioritizes a clear correctness baseline. It
     * enumerates the full index space and is not intended to replace optimized
     * backend kernels for large contractions.
     *
     * @param equation explicit einsum equation, for example {@code ij,jk->ik}
     * @param tensors tensors in equation operand order
     * @return a new tensor with the dimensions named by the output labels
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
            if (tensors[i].dtype() != DType.F32 || tensors[i].device() != DeviceType.CPU) {
                throw new UnsupportedOperationException("einsum currently supports F32 CPU tensors only");
            }
        }

        String normalized = equation.replaceAll("\\s+", "");
        String[] sides = normalized.split("->", -1);
        if (sides.length != 2) {
            throw new IllegalArgumentException("einsum requires exactly one explicit '->' output");
        }
        String[] inputLabels = sides[0].split(",", -1);
        String outputLabels = sides[1];
        if (inputLabels.length != tensors.length) {
            throw new IllegalArgumentException("equation has " + inputLabels.length
                    + " operands but received " + tensors.length + " tensors");
        }

        int[] labelDimensions = new int[LABEL_COUNT];
        boolean[] labelSeen = new boolean[LABEL_COUNT];
        List<Integer> labelsInOrder = new ArrayList<>(LABEL_COUNT);
        for (int operand = 0; operand < tensors.length; operand++) {
            String labels = inputLabels[operand];
            if (labels.length() != tensors[operand].shape().rank()) {
                throw new IllegalArgumentException("operand " + operand + " has rank "
                        + tensors[operand].shape().rank() + " but equation labels '" + labels
                        + "' have length " + labels.length());
            }
            boolean[] localSeen = new boolean[LABEL_COUNT];
            for (int axis = 0; axis < labels.length(); axis++) {
                int slot = labelSlot(labels.charAt(axis));
                if (localSeen[slot]) {
                    throw new IllegalArgumentException("repeated label '" + labels.charAt(axis)
                            + "' within operand " + operand + " is not supported");
                }
                localSeen[slot] = true;
                int dimension = Math.toIntExact(tensors[operand].shape().dim(axis));
                if (labelSeen[slot] && labelDimensions[slot] != dimension) {
                    throw new IllegalArgumentException("dimension mismatch for label '" + labels.charAt(axis)
                            + "': expected " + labelDimensions[slot] + " but got " + dimension);
                }
                if (!labelSeen[slot]) {
                    labelSeen[slot] = true;
                    labelDimensions[slot] = dimension;
                    labelsInOrder.add(slot);
                }
            }
        }

        boolean[] outputSeen = new boolean[LABEL_COUNT];
        long[] outputDims = new long[outputLabels.length()];
        for (int axis = 0; axis < outputLabels.length(); axis++) {
            int slot = labelSlot(outputLabels.charAt(axis));
            if (outputSeen[slot]) {
                throw new IllegalArgumentException("repeated output label '" + outputLabels.charAt(axis) + "'");
            }
            if (!labelSeen[slot]) {
                throw new IllegalArgumentException("output label '" + outputLabels.charAt(axis)
                        + "' does not occur in an input");
            }
            outputSeen[slot] = true;
            outputDims[axis] = labelDimensions[slot];
        }

        long outputElements = new Shape(outputDims).numel();
        if (outputElements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("einsum output is too large for a Java float[]: " + outputElements);
        }

        long assignmentCount = 1;
        try {
            for (int slot : labelsInOrder) {
                assignmentCount = Math.multiplyExact(assignmentCount, labelDimensions[slot]);
            }
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("einsum index space overflows long", e);
        }
        if (assignmentCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("einsum index space is too large for the reference CPU implementation: "
                    + assignmentCount);
        }

        float[][] values = new float[tensors.length][];
        for (int i = 0; i < tensors.length; i++) {
            values[i] = tensors[i].toFloatArray();
        }
        float[] output = new float[Math.toIntExact(outputElements)];
        int[] coordinates = new int[LABEL_COUNT];
        for (long assignment = 0; assignment < assignmentCount; assignment++) {
            long remainder = assignment;
            for (int i = labelsInOrder.size() - 1; i >= 0; i--) {
                int slot = labelsInOrder.get(i);
                int dimension = labelDimensions[slot];
                coordinates[slot] = dimension == 0 ? 0 : (int) (remainder % dimension);
                if (dimension != 0) {
                    remainder /= dimension;
                }
            }

            float product = 1.0f;
            for (int operand = 0; operand < tensors.length; operand++) {
                String labels = inputLabels[operand];
                long offset = 0;
                for (int axis = 0; axis < labels.length(); axis++) {
                    int slot = labelSlot(labels.charAt(axis));
                    offset = offset * labelDimensions[slot] + coordinates[slot];
                }
                product *= values[operand][Math.toIntExact(offset)];
            }

            long outputOffset = 0;
            for (int axis = 0; axis < outputLabels.length(); axis++) {
                int slot = labelSlot(outputLabels.charAt(axis));
                outputOffset = outputOffset * labelDimensions[slot] + coordinates[slot];
            }
            output[Math.toIntExact(outputOffset)] += product;
        }
        return Tensor.of(output, outputDims);
    }

    private static int labelSlot(char label) {
        if (label >= 'a' && label <= 'z') {
            return label - 'a';
        }
        if (label >= 'A' && label <= 'Z') {
            return 26 + label - 'A';
        }
        throw new IllegalArgumentException("einsum labels must be ASCII letters; found '" + label + "'");
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
        try {
            for (int d = 0; d < axis; d++) {
                outer = Math.multiplyExact(outer, first.shape().dim(d));
            }
            for (int d = axis + 1; d < rank; d++) {
                inner = Math.multiplyExact(inner, first.shape().dim(d));
            }
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("cat block size overflows long", e);
        }

        float[][] sources = new float[tensors.size()][];
        for (int i = 0; i < tensors.size(); i++) {
            sources[i] = tensors.get(i).toFloatArray();
        }

        float[] output = new float[Math.toIntExact(outputElements)];
        int destination = 0;
        for (long outerIndex = 0; outerIndex < outer; outerIndex++) {
            for (int i = 0; i < tensors.size(); i++) {
                Tensor tensor = tensors.get(i);
                long blockElements = Math.multiplyExact(tensor.shape().dim(axis), inner);
                long sourceOffset = Math.multiplyExact(outerIndex, blockElements);
                int blockLength = Math.toIntExact(blockElements);
                System.arraycopy(sources[i], Math.toIntExact(sourceOffset), output, destination, blockLength);
                destination += blockLength;
            }
        }
        return Tensor.of(output, outputDims);
    }
}
