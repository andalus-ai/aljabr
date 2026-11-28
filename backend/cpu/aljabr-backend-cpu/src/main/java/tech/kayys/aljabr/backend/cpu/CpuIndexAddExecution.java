package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Shape;
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
        if (!isSupported(dtype)) {
            throw new UnsupportedOperationException(
                    "CPU indexAdd currently supports F32, F16, BF16, I32, I8, and INT8 destination and source tensors; actual dtype="
                            + dtype);
        }
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

            copyDestination(dtype, destination, destinationDimensions, destinationLayout,
                    result, outputElements);

            long sourceAxisSize = sourceDimensions[selectedAxis];
            for (long outerIndex = 0; outerIndex < outer; outerIndex++) {
                for (int indexPosition = 0; indexPosition < selected.length; indexPosition++) {
                    long sourceBase = Math.multiplyExact(
                            Math.addExact(Math.multiplyExact(outerIndex, sourceAxisSize), indexPosition), inner);
                    long destinationBase = Math.multiplyExact(
                            Math.addExact(Math.multiplyExact(outerIndex, destinationAxisSize), selected[indexPosition]), inner);
                    for (long innerIndex = 0; innerIndex < inner; innerIndex++) {
                        long sourceLogical = Math.addExact(sourceBase, innerIndex);
                        long destinationLogical = Math.addExact(destinationBase, innerIndex);
                        accumulate(dtype, updates, sourceDimensions, sourceLayout,
                                result, destinationDimensions, resultLayout,
                                sourceLogical, destinationLogical);
                    }
                }
            }
            return backend.out(plan.outputShape(), dtype, output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }

    private static boolean isSupported(DType dtype) {
        return dtype == DType.F32 || dtype == DType.F16 || dtype == DType.BF16
                || dtype == DType.I32 || dtype == DType.I8 || dtype == DType.INT8;
    }

    private static void copyDestination(
            DType dtype, MemorySegment source, long[] dimensions, TensorLayout layout,
            MemorySegment destination, long elements) {
        for (long logical = 0; logical < elements; logical++) {
            if (dtype == DType.I32) {
                CpuIndexAddValueAccess.writeIndexAddInt(destination, logical,
                        CpuIndexAddValueAccess.readIndexAddInt(source, dimensions, layout, logical));
            } else if (dtype == DType.I8 || dtype == DType.INT8) {
                CpuIndexAddValueAccess.writeIndexAddByte(destination, logical,
                        CpuIndexAddValueAccess.readIndexAddByte(source, dimensions, layout, logical));
            } else {
                float value = CpuIndexAddValueAccess.readIndexAddValue(source, dimensions, layout, logical, dtype);
                CpuIndexAddValueAccess.writeIndexAddValue(destination, logical, dtype, value);
            }
        }
    }

    private static void accumulate(
            DType dtype,
            MemorySegment updates, long[] sourceDimensions, TensorLayout sourceLayout,
            MemorySegment result, long[] destinationDimensions, TensorLayout resultLayout,
            long sourceLogical, long destinationLogical) {
        if (dtype == DType.I32) {
            int update = CpuIndexAddValueAccess.readIndexAddInt(
                    updates, sourceDimensions, sourceLayout, sourceLogical);
            int current = CpuIndexAddValueAccess.readIndexAddInt(
                    result, destinationDimensions, resultLayout, destinationLogical);
            CpuIndexAddValueAccess.writeIndexAddInt(result, destinationLogical, current + update);
        } else if (dtype == DType.I8 || dtype == DType.INT8) {
            byte update = CpuIndexAddValueAccess.readIndexAddByte(
                    updates, sourceDimensions, sourceLayout, sourceLogical);
            byte current = CpuIndexAddValueAccess.readIndexAddByte(
                    result, destinationDimensions, resultLayout, destinationLogical);
            CpuIndexAddValueAccess.writeIndexAddByte(result, destinationLogical, (byte) (current + update));
        } else {
            float update = CpuIndexAddValueAccess.readIndexAddValue(
                    updates, sourceDimensions, sourceLayout, sourceLogical, dtype);
            float current = CpuIndexAddValueAccess.readIndexAddValue(
                    result, destinationDimensions, resultLayout, destinationLogical, dtype);
            CpuIndexAddValueAccess.writeIndexAddValue(result, destinationLogical, dtype, current + update);
        }
    }
}
