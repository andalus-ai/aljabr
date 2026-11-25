package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.LongFunction;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/**
 * CPU layer, RMS and batch normalization execution.
 *
 * <p>This component owns normalization validation and numerical execution;
 * the backend facade retains output allocation policy and tensor construction.</p>
 */
final class CpuNormalizationExecution {
    @FunctionalInterface
    interface OutputFactory {
        Tensor create(Shape shape, HostAccessibleStorage storage);
    }

    private final LongFunction<HostAccessibleStorage> outputAllocator;
    private final OutputFactory outputFactory;

    CpuNormalizationExecution(LongFunction<HostAccessibleStorage> outputAllocator, OutputFactory outputFactory) {
        this.outputAllocator = Objects.requireNonNull(outputAllocator, "outputAllocator");
        this.outputFactory = Objects.requireNonNull(outputFactory, "outputFactory");
    }

    private static MemorySegment seg(Tensor tensor) {
        return CpuTensorStorageAccess.segment(tensor);
    }

    public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(normalizedShape, "normalizedShape");
        CpuTensorLayoutAccess.requireF32(input);
        requireNormalizationEpsilon(eps);
        Shape shape = input.shape();
        long[] dimensions = shape.dims();
        if (normalizedShape.length == 0 || normalizedShape.length > dimensions.length) {
            throw new IllegalArgumentException("normalizedShape must be a non-empty suffix of the input shape");
        }
        long hiddenDim = 1;
        for (int i = 0; i < normalizedShape.length; i++) {
            long expected = normalizedShape[i];
            long actual = dimensions[dimensions.length - normalizedShape.length + i];
            if (expected <= 0 || expected != actual) {
                throw new IllegalArgumentException("normalizedShape must match the trailing input dimensions");
            }
            hiddenDim = Math.multiplyExact(hiddenDim, expected);
        }
        validateNormParameter(weight, hiddenDim, "weight");
        validateNormParameter(bias, hiddenDim, "bias");
        long elements = shape.numel();
        HostAccessibleStorage output = outputAllocator.apply(Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) return outputFactory.create(shape, output);

        MemorySegment source = seg(input), destination = output.segment();
        MemorySegment scale = weight == null ? null : seg(weight);
        MemorySegment shift = bias == null ? null : seg(bias);
        TensorLayout layout = input.layout();
        long rows = elements / hiddenDim;
        for (long row = 0; row < rows; row++) {
            long rowStart = Math.multiplyExact(row, hiddenDim);
            double sum = 0d;
            for (long column = 0; column < hiddenDim; column++) {
                sum += CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                        Math.addExact(rowStart, column));
            }
            double mean = sum / hiddenDim;
            double squaredDifferenceSum = 0d;
            for (long column = 0; column < hiddenDim; column++) {
                double difference = CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                        Math.addExact(rowStart, column)) - mean;
                squaredDifferenceSum += difference * difference;
            }
            float inverseStd = (float) (1d / Math.sqrt(squaredDifferenceSum / hiddenDim + eps));
            for (long column = 0; column < hiddenDim; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                float normalized = (value - (float) mean) * inverseStd;
                if (scale != null) normalized *= scale.get(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(column, Float.BYTES));
                if (shift != null) normalized += shift.get(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(column, Float.BYTES));
                destination.set(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(logicalIndex, Float.BYTES), normalized);
            }
        }
        return outputFactory.create(shape, output);
    }

    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        CpuTensorLayoutAccess.requireF32(input);
        CpuTensorLayoutAccess.requireF32(weight);
        requireNormalizationEpsilon(eps);
        if (weight.shape().rank() != 1) {
            throw new IllegalArgumentException("RMSNorm weight must be a rank-1 vector");
        }
        long hiddenDim = weight.shape().dim(0);
        if (hiddenDim <= 0 || input.shape().rank() == 0
                || input.shape().dim(input.shape().rank() - 1) != hiddenDim) {
            throw new IllegalArgumentException("RMSNorm weight length must match the input's final dimension");
        }
        validateNormParameter(weight, hiddenDim, "weight");
        long elements = input.numel();
        HostAccessibleStorage output = outputAllocator.apply(Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) return outputFactory.create(input.shape(), output);

        MemorySegment source = seg(input), scale = seg(weight), destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();
        long rows = elements / hiddenDim;
        for (long row = 0; row < rows; row++) {
            long rowStart = Math.multiplyExact(row, hiddenDim);
            double sumSquares = 0d;
            for (long column = 0; column < hiddenDim; column++) {
                float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                        Math.addExact(rowStart, column));
                sumSquares += (double) value * value;
            }
            float inverseRms = (float) (1d / Math.sqrt(sumSquares / hiddenDim + eps));
            for (long column = 0; column < hiddenDim; column++) {
                long logicalIndex = Math.addExact(rowStart, column);
                float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                float gamma = scale.get(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(column, Float.BYTES));
                destination.set(ValueLayout.JAVA_FLOAT,
                        Math.multiplyExact(logicalIndex, Float.BYTES), (value * inverseRms) * gamma);
            }
        }
        return outputFactory.create(input.shape(), output);
    }

    private static void validateNormParameter(Tensor parameter, long expectedElements, String name) {
        if (parameter == null) return;
        CpuTensorLayoutAccess.requireF32(parameter);
        if (parameter.numel() != expectedElements) {
            throw new IllegalArgumentException(name + " must contain " + expectedElements
                    + " elements; got " + parameter.numel());
        }
        TensorLayout layout = parameter.layout();
        if (!layout.isContiguous() || layout.storageOffset() != 0) {
            throw new IllegalArgumentException(name + " must use contiguous zero-offset storage");
        }
    }

    private static void requireNormalizationEpsilon(float eps) {
        if (!Float.isFinite(eps) || eps < 0f) {
            throw new IllegalArgumentException("Normalization epsilon must be finite and non-negative");
        }
    }

    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias,
            Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        Objects.requireNonNull(input, "input");
        CpuTensorLayoutAccess.requireF32(input);
        if (input.shape().rank() < 2) {
            throw new IllegalArgumentException("BatchNorm input must have rank >= 2 ([N, C, ...])");
        }
        if (!Float.isFinite(momentum) || momentum < 0f || momentum > 1f) {
            throw new IllegalArgumentException("BatchNorm momentum must be finite and in [0, 1]");
        }
        if (!Float.isFinite(eps) || eps < 0f) {
            throw new IllegalArgumentException("BatchNorm epsilon must be finite and non-negative");
        }

        long[] dimensions = input.shape().dims();
        long batches = dimensions[0];
        long channels = dimensions[1];
        if (channels <= 0) {
            throw new IllegalArgumentException("BatchNorm channel dimension must be positive");
        }
        long spatialSize = 1;
        for (int i = 2; i < dimensions.length; i++) {
            spatialSize = Math.multiplyExact(spatialSize, dimensions[i]);
        }
        long samplesPerChannel = Math.multiplyExact(batches, spatialSize);
        validateBatchNormVector(weight, channels, "weight");
        validateBatchNormVector(bias, channels, "bias");
        if ((runningMean == null) != (runningVar == null)) {
            throw new IllegalArgumentException("runningMean and runningVar must either both be present or both be null");
        }
        validateBatchNormVector(runningMean, channels, "runningMean");
        validateBatchNormVector(runningVar, channels, "runningVar");
        long elements = input.numel();
        HostAccessibleStorage output = outputAllocator.apply(Math.multiplyExact(elements, (long) Float.BYTES));
        if (elements == 0) return outputFactory.create(input.shape(), output);

        MemorySegment source = seg(input);
        MemorySegment destination = output.segment();
        MemorySegment scale = weight == null ? null : seg(weight);
        MemorySegment shift = bias == null ? null : seg(bias);
        MemorySegment means = runningMean == null ? null : seg(runningMean);
        MemorySegment variances = runningVar == null ? null : seg(runningVar);
        TensorLayout layout = input.layout();

        for (long channel = 0; channel < channels; channel++) {
            double mean;
            double variance;
            if (training || means == null) {
                double sum = 0d;
                double sumSquares = 0d;
                for (long batch = 0; batch < batches; batch++) {
                    long base = Math.multiplyExact(
                            Math.addExact(Math.multiplyExact(batch, channels), channel), spatialSize);
                    for (long spatial = 0; spatial < spatialSize; spatial++) {
                        float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout,
                                Math.addExact(base, spatial));
                        sum += value;
                        sumSquares += (double) value * value;
                    }
                }
                mean = sum / samplesPerChannel;
                variance = Math.max(0d, sumSquares / samplesPerChannel - mean * mean);

                if (training && means != null) {
                    double runningVariance = samplesPerChannel > 1
                            ? variance * samplesPerChannel / (samplesPerChannel - 1d)
                            : variance;
                    long parameterOffset = Math.multiplyExact(channel, Float.BYTES);
                    float oldMean = means.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                    float oldVariance = variances.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                    if (!Float.isFinite(oldMean) || !Float.isFinite(oldVariance) || oldVariance < 0f) {
                        throw new IllegalArgumentException(
                                "BatchNorm running statistics must be finite and variance non-negative");
                    }
                    means.set(ValueLayout.JAVA_FLOAT, parameterOffset,
                            (float) ((1d - momentum) * oldMean + momentum * mean));
                    variances.set(ValueLayout.JAVA_FLOAT, parameterOffset,
                            (float) ((1d - momentum) * oldVariance + momentum * runningVariance));
                }
            } else {
                long parameterOffset = Math.multiplyExact(channel, Float.BYTES);
                mean = means.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                variance = variances.get(ValueLayout.JAVA_FLOAT, parameterOffset);
                if (!Double.isFinite(mean) || !Double.isFinite(variance) || variance < 0d) {
                    throw new IllegalArgumentException("BatchNorm running statistics must be finite and variance non-negative");
                }
            }

            float inverseStd = (float) (1d / Math.sqrt(variance + eps));
            float gamma = scale == null ? 1f
                    : scale.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(channel, Float.BYTES));
            float beta = shift == null ? 0f
                    : shift.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(channel, Float.BYTES));
            for (long batch = 0; batch < batches; batch++) {
                long base = Math.multiplyExact(
                        Math.addExact(Math.multiplyExact(batch, channels), channel), spatialSize);
                for (long spatial = 0; spatial < spatialSize; spatial++) {
                    long logicalIndex = Math.addExact(base, spatial);
                    float value = CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                    float normalized = (float) ((value - mean) * inverseStd) * gamma + beta;
                    destination.set(ValueLayout.JAVA_FLOAT,
                            Math.multiplyExact(logicalIndex, Float.BYTES), normalized);
                }
            }
        }
        return outputFactory.create(input.shape(), output);
    }


    private static void validateBatchNormVector(Tensor tensor, long channels, String name) {
        if (tensor == null) return;
        CpuTensorLayoutAccess.requireF32(tensor);
        if (tensor.shape().rank() != 1 || tensor.shape().dim(0) != channels) {
            throw new IllegalArgumentException(name + " must be a rank-1 vector of length " + channels);
        }
        TensorLayout layout = tensor.layout();
        if (!layout.isContiguous() || layout.storageOffset() != 0) {
            throw new IllegalArgumentException(name + " must use contiguous zero-offset storage");
        }
    }
}
