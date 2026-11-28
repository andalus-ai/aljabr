package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;
import tech.kayys.aljabr.core.tensor.internal.TensorOperationValidator;

/**
 * Owns CPU indexAdd validation, layout-aware traversal, and output lifecycle.
 *
 * <p>Storage representation is delegated to {@link CpuIndexAddValueAccess};
 * backend allocation and tensor construction remain owned by {@link CpuBackend}.
 */
final class CpuIndexAddExecution {
    private final CpuBackend backend;

    CpuIndexAddExecution(CpuBackend backend) {
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    Tensor indexAdd(Tensor input, int axis, Tensor indices, Tensor source) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(indices, "indices");
        Objects.requireNonNull(source, "source");
        TensorOperationValidator.IndexAddPlan plan = TensorOperationValidator.validateIndexAdd(
                input.shape(), indices.shape(), source.shape(), axis);
        if (input.backend() != backend || indices.backend() != backend || source.backend() != backend) {
            throw new IllegalArgumentException("CPU indexAdd requires all tensors to use this backend");
        }
        DType dtype = input.dtype();
        if (dtype != source.dtype()) {
            throw new IllegalArgumentException(
                    "CPU indexAdd source dtype " + source.dtype()
                            + " must match destination dtype " + dtype);
        }
        CpuIndexAddArithmetic arithmetic = CpuIndexAddArithmetic.forDType(dtype);
        if (indices.dtype() != DType.I32) {
            throw new IllegalArgumentException("CPU indexAdd indices must use I32; actual dtype=" + indices.dtype());
        }

        int[] selected = indices.toIntArray();
        int selectedAxis = plan.axis();
        long destinationAxisSize = input.shape().dim(selectedAxis);
        for (int i = 0; i < selected.length; i++) {
            if (selected[i] < 0 || selected[i] >= destinationAxisSize) {
                throw new IndexOutOfBoundsException(
                        "index-add index " + selected[i] + " at position " + i
                                + " is outside axis " + selectedAxis + " with size " + destinationAxisSize);
            }
        }

        long[] destinationDimensions = input.shape().dims();
        long[] sourceDimensions = source.shape().dims();
        long outer = 1;
        long inner = 1;
        for (int i = 0; i < selectedAxis; i++) {
            outer = Math.multiplyExact(outer, destinationDimensions[i]);
        }
        for (int i = selectedAxis + 1; i < destinationDimensions.length; i++) {
            inner = Math.multiplyExact(inner, destinationDimensions[i]);
        }

        long outputElements = plan.outputShape().numel();
        HostAccessibleStorage output = backend.allocateOutput(dtype.memoryFootprintBytes(outputElements));
        try {
            MemorySegment destination = CpuTensorStorageAccess.segment(input);
            MemorySegment updates = CpuTensorStorageAccess.segment(source);
            MemorySegment result = output.segment();
            TensorLayout destinationLayout = input.layout();
            TensorLayout sourceLayout = source.layout();
            TensorLayout resultLayout = TensorLayout.contiguous(plan.outputShape());

            for (long logical = 0; logical < outputElements; logical++) {
                arithmetic.copy(destination, destinationDimensions, destinationLayout, result, logical);
            }

            long sourceAxisSize = sourceDimensions[selectedAxis];
            CpuIndexedAxisTraversal.forEach(outer, sourceAxisSize, inner, selected,
                    (outerIndex, selectionPosition, selectedIndex, innerIndex, sourceLogical, outputLogical) -> {
                        long destinationLogical = Math.addExact(
                                Math.multiplyExact(
                                        Math.addExact(Math.multiplyExact(outerIndex, destinationAxisSize), selectedIndex),
                                        inner),
                                innerIndex);
                        arithmetic.add(updates, sourceDimensions, sourceLayout,
                                result, destinationDimensions, resultLayout,
                                sourceLogical, destinationLogical);
                    });
            return backend.out(plan.outputShape(), dtype, output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }


}
