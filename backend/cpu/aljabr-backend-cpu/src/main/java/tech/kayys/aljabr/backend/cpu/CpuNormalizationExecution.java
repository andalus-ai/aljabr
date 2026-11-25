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
        return normalizationExecution.layerNorm(input, normalizedShape, weight, bias, eps);
    }

    public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        return normalizationExecution.rmsNorm(input, weight, eps);
    }

    public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias,
            Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        return normalizationExecution.batchNorm(
                input, weight, bias, runningMean, runningVar, training, momentum, eps);
    }

}
