package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/**
 * Logical element access for CPU pointwise kernels.
 *
 * <p>Contiguous zero-offset tensors retain the SIMD implementation in
 * {@code CpuOps}. Strided and offset tensors are traversed in logical
 * row-major order and gathered through their element strides.</p>
 */
final class CpuTensorLayoutAccess {
    @FunctionalInterface
    interface ContiguousBinaryKernel {
        void apply(MemorySegment left, MemorySegment right, MemorySegment output, long elements);
    }

    @FunctionalInterface
    interface ContiguousUnaryKernel {
        void apply(MemorySegment input, MemorySegment output, long elements);
    }

    @FunctionalInterface
    interface BinaryFloatOperator {
        float apply(float left, float right);
    }

    @FunctionalInterface
    interface UnaryFloatOperator {
        float apply(float value);
    }

    private CpuTensorLayoutAccess() {
    }

    static void binary(
            Tensor left, Tensor right, MemorySegment output, long elements,
            ContiguousBinaryKernel contiguousKernel, BinaryFloatOperator operator) {
        requireF32(left);
        requireF32(right);
        Objects.requireNonNull(operator, "operator");
        MemorySegment leftSegment = CpuTensorStorageAccess.segment(left);
        MemorySegment rightSegment = CpuTensorStorageAccess.segment(right);
        TensorLayout leftLayout = left.layout();
        TensorLayout rightLayout = right.layout();
        if (leftLayout.isContiguous() && leftLayout.storageOffset() == 0
                && rightLayout.isContiguous() && rightLayout.storageOffset() == 0) {
            contiguousKernel.apply(leftSegment, rightSegment, output, elements);
            return;
        }
        for (long i = 0; i < elements; i++) {
            float a = readLogical(leftSegment, left.shape().dims(), leftLayout, i);
            float b = readLogical(rightSegment, right.shape().dims(), rightLayout, i);
            output.set(ValueLayout.JAVA_FLOAT, i * Float.BYTES, operator.apply(a, b));
        }
    }

    static void unary(Tensor input, MemorySegment output, long elements,
            ContiguousUnaryKernel contiguousKernel, UnaryFloatOperator operator) {
        requireF32(input);
        Objects.requireNonNull(operator, "operator");
        MemorySegment source = CpuTensorStorageAccess.segment(input);
        TensorLayout layout = input.layout();
        if (layout.isContiguous() && layout.storageOffset() == 0) {
            contiguousKernel.apply(source, output, elements);
            return;
        }
        long[] dimensions = input.shape().dims();
        for (long i = 0; i < elements; i++) {
            float value = readLogical(source, dimensions, layout, i);
            output.set(ValueLayout.JAVA_FLOAT, i * Float.BYTES, operator.apply(value));
        }
    }

    static MemorySegment contiguousF32(Tensor tensor) {
        requireF32(tensor);
        MemorySegment source = CpuTensorStorageAccess.segment(tensor);
        TensorLayout layout = tensor.layout();
        if (layout.isContiguous() && layout.storageOffset() == 0) {
            return source;
        }
        long elements = tensor.numel();
        float[] values = new float[Math.toIntExact(elements)];
        long[] dimensions = tensor.shape().dims();
        for (int index = 0; index < values.length; index++) {
            values[index] = readLogical(source, dimensions, layout, index);
        }
        return MemorySegment.ofArray(values);
    }

    static float readLogical(
            MemorySegment source, long[] dimensions, TensorLayout layout, long logicalIndex) {
        long[] strides = layout.strides();
        long remaining = logicalIndex;
        long elementOffset = layout.storageOffset();
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            long coordinate = remaining % dimensions[axis];
            remaining /= dimensions[axis];
            elementOffset = Math.addExact(elementOffset, Math.multiplyExact(coordinate, strides[axis]));
        }
        return source.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(elementOffset, Float.BYTES));
    }

    static void requireF32(Tensor tensor) {
        if (tensor.dtype() != DType.F32) {
            throw new UnsupportedOperationException(
                    "CPU tensor kernels currently support F32 tensors only; actual dtype=" + tensor.dtype());
        }
    }
}
