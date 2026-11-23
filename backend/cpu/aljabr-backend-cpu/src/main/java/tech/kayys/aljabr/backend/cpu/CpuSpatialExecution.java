package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.LongFunction;

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;

/** CPU NCHW convolution, pooling and nearest-neighbor upsampling execution. */
final class CpuSpatialExecution {
    @FunctionalInterface
    interface OutputFactory {
        Tensor create(Shape shape, HostAccessibleStorage storage);
    }

    private final LongFunction<HostAccessibleStorage> outputAllocator;
    private final OutputFactory outputFactory;

    CpuSpatialExecution(LongFunction<HostAccessibleStorage> outputAllocator, OutputFactory outputFactory) {
        this.outputAllocator = Objects.requireNonNull(outputAllocator, "outputAllocator");
        this.outputFactory = Objects.requireNonNull(outputFactory, "outputFactory");
    }

    Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
            int stride, int padding, int dilation, int groups) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        if (input.shape().rank() != 4 || weight.shape().rank() != 4) {
            throw new IllegalArgumentException("conv2d expects rank-4 NCHW input and OIHW weight");
        }
        CpuTensorLayoutAccess.requireF32(input);
        CpuTensorLayoutAccess.requireF32(weight);
        if (bias != null) {
            if (bias.shape().rank() != 1) {
                throw new IllegalArgumentException("conv2d bias must be a rank-1 [C_out] tensor");
            }
            CpuTensorLayoutAccess.requireF32(bias);
        }
        if (stride <= 0 || dilation <= 0 || padding < 0 || groups <= 0) {
            throw new IllegalArgumentException(
                    "stride and dilation must be positive; padding must be non-negative; groups must be positive");
        }

        long batch = input.shape().dim(0);
        long inputChannels = input.shape().dim(1);
        long inputHeight = input.shape().dim(2);
        long inputWidth = input.shape().dim(3);
        long outputChannels = weight.shape().dim(0);
        long channelsPerGroup = weight.shape().dim(1);
        long kernelHeight = weight.shape().dim(2);
        long kernelWidth = weight.shape().dim(3);

        if (inputChannels <= 0 || inputHeight <= 0 || inputWidth <= 0
                || outputChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0) {
            throw new IllegalArgumentException("conv2d channel and spatial dimensions must be positive");
        }
        if (inputChannels % groups != 0 || outputChannels % groups != 0
                || channelsPerGroup != inputChannels / groups) {
            throw new IllegalArgumentException(
                    "conv2d groups must divide input/output channels and weight shape must be [C_out, C_in/groups, kH, kW]");
        }
        if (bias != null && bias.shape().dim(0) != outputChannels) {
            throw new IllegalArgumentException("conv2d bias length must equal output channels");
        }

        long effectiveKernelHeight = Math.addExact(Math.multiplyExact((long) dilation, kernelHeight - 1), 1L);
        long effectiveKernelWidth = Math.addExact(Math.multiplyExact((long) dilation, kernelWidth - 1), 1L);
        long paddedHeight = Math.addExact(inputHeight, Math.multiplyExact(2L, padding));
        long paddedWidth = Math.addExact(inputWidth, Math.multiplyExact(2L, padding));
        if (effectiveKernelHeight > paddedHeight || effectiveKernelWidth > paddedWidth) {
            throw new IllegalArgumentException("conv2d effective kernel exceeds padded input dimensions");
        }
        long outputHeight = (paddedHeight - effectiveKernelHeight) / stride + 1;
        long outputWidth = (paddedWidth - effectiveKernelWidth) / stride + 1;
        Shape outputShape = new Shape(batch, outputChannels, outputHeight, outputWidth);
        long outputBytes = Math.multiplyExact(outputShape.numel(), Float.BYTES);
        HostAccessibleStorage output = outputAllocator.apply(outputBytes);

        MemorySegment inputSegment = CpuTensorStorageAccess.segment(input);
        MemorySegment weightSegment = CpuTensorStorageAccess.segment(weight);
        MemorySegment biasSegment = bias == null ? null : CpuTensorStorageAccess.segment(bias);
        MemorySegment outputSegment = output.segment();
        long[] inputDimensions = input.shape().dims();
        long[] weightDimensions = weight.shape().dims();
        long[] biasDimensions = bias == null ? null : bias.shape().dims();
        TensorLayout inputLayout = input.layout();
        TensorLayout weightLayout = weight.layout();
        TensorLayout biasLayout = bias == null ? null : bias.layout();
        long outputChannelsPerGroup = outputChannels / groups;

        for (long n = 0; n < batch; n++) {
            for (long outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
                long group = outputChannel / outputChannelsPerGroup;
                long inputChannelStart = group * channelsPerGroup;
                float biasValue = bias == null ? 0f : CpuTensorLayoutAccess.readLogical(
                        biasSegment, biasDimensions, biasLayout, outputChannel);
                for (long outputY = 0; outputY < outputHeight; outputY++) {
                    for (long outputX = 0; outputX < outputWidth; outputX++) {
                        float accumulator = biasValue;
                        for (long inputChannelInGroup = 0;
                                inputChannelInGroup < channelsPerGroup; inputChannelInGroup++) {
                            long inputChannel = inputChannelStart + inputChannelInGroup;
                            for (long kernelY = 0; kernelY < kernelHeight; kernelY++) {
                                long inputY = outputY * stride - padding + kernelY * dilation;
                                if (inputY < 0 || inputY >= inputHeight) continue;
                                for (long kernelX = 0; kernelX < kernelWidth; kernelX++) {
                                    long inputX = outputX * stride - padding + kernelX * dilation;
                                    if (inputX < 0 || inputX >= inputWidth) continue;
                                    long inputIndex = ((n * inputChannels + inputChannel)
                                            * inputHeight + inputY) * inputWidth + inputX;
                                    long weightIndex = ((outputChannel * channelsPerGroup
                                            + inputChannelInGroup) * kernelHeight + kernelY)
                                            * kernelWidth + kernelX;
                                    float inputValue = CpuTensorLayoutAccess.readLogical(
                                            inputSegment, inputDimensions, inputLayout, inputIndex);
                                    float weightValue = CpuTensorLayoutAccess.readLogical(
                                            weightSegment, weightDimensions, weightLayout, weightIndex);
                                    accumulator += inputValue * weightValue;
                                }
                            }
                        }
                        long outputIndex = ((n * outputChannels + outputChannel)
                                * outputHeight + outputY) * outputWidth + outputX;
                        outputSegment.set(ValueLayout.JAVA_FLOAT,
                                outputIndex * Float.BYTES, accumulator);
                    }
                }
            }
        }
        return outputFactory.create(outputShape, output);
    }

    Tensor upsampleNearest2d(Tensor input, int outputH, int outputW) {
        requireSpatialInput(input, "upsampleNearest2d");
        if (outputH <= 0 || outputW <= 0) {
            throw new IllegalArgumentException("output height and width must be positive");
        }
        long batch = input.shape().dim(0);
        long channels = input.shape().dim(1);
        long inputH = input.shape().dim(2);
        long inputW = input.shape().dim(3);
        if (inputH <= 0 || inputW <= 0) {
            throw new IllegalArgumentException("input height and width must be positive");
        }

        Shape outputShape = new Shape(batch, channels, outputH, outputW);
        HostAccessibleStorage output = outputAllocator.apply(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment source = CpuTensorStorageAccess.segment(input);
        MemorySegment destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();

        for (long n = 0; n < batch; n++) {
            for (long c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputH; oh++) {
                    long ih = Math.min(inputH - 1, (long) oh * inputH / outputH);
                    for (int ow = 0; ow < outputW; ow++) {
                        long iw = Math.min(inputW - 1, (long) ow * inputW / outputW);
                        long logicalIndex = ((n * channels + c) * inputH + ih) * inputW + iw;
                        long destinationIndex = ((n * channels + c) * outputH + oh) * outputW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, destinationIndex * Float.BYTES,
                                CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex));
                    }
                }
            }
        }
        return outputFactory.create(outputShape, output);
    }

    Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        requireSpatialInput(input, "maxPool2d");
        if (kernelSize <= 0 || stride <= 0 || padding < 0) {
            throw new IllegalArgumentException("kernel size and stride must be positive; padding must be non-negative");
        }
        long nSize = input.shape().dim(0);
        long channels = input.shape().dim(1);
        long height = input.shape().dim(2);
        long width = input.shape().dim(3);
        long outH = (height + 2L * padding - kernelSize) / stride + 1;
        long outW = (width + 2L * padding - kernelSize) / stride + 1;
        if (outH <= 0 || outW <= 0) {
            throw new IllegalArgumentException("pooling kernel produces an empty spatial output");
        }

        Shape outputShape = new Shape(nSize, channels, outH, outW);
        HostAccessibleStorage output = outputAllocator.apply(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment source = CpuTensorStorageAccess.segment(input);
        MemorySegment destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();

        for (long n = 0; n < nSize; n++) {
            for (long c = 0; c < channels; c++) {
                for (long oh = 0; oh < outH; oh++) {
                    for (long ow = 0; ow < outW; ow++) {
                        float best = Float.NEGATIVE_INFINITY;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                long ih = oh * stride - padding + kh;
                                long iw = ow * stride - padding + kw;
                                if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;
                                long logicalIndex = ((n * channels + c) * height + ih) * width + iw;
                                float value = CpuTensorLayoutAccess.readLogical(
                                        source, dimensions, layout, logicalIndex);
                                if (value > best) best = value;
                            }
                        }
                        long outputIndex = ((n * channels + c) * outH + oh) * outW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, outputIndex * Float.BYTES, best);
                    }
                }
            }
        }
        return outputFactory.create(outputShape, output);
    }

    Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        requireSpatialInput(input, "adaptiveAvgPool2d");
        if (outputH <= 0 || outputW <= 0) {
            throw new IllegalArgumentException("output height and width must be positive");
        }
        long nSize = input.shape().dim(0);
        long channels = input.shape().dim(1);
        long height = input.shape().dim(2);
        long width = input.shape().dim(3);
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("input height and width must be positive");
        }

        Shape outputShape = new Shape(nSize, channels, outputH, outputW);
        HostAccessibleStorage output = outputAllocator.apply(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment source = CpuTensorStorageAccess.segment(input);
        MemorySegment destination = output.segment();
        long[] dimensions = input.shape().dims();
        TensorLayout layout = input.layout();

        for (long n = 0; n < nSize; n++) {
            for (long c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputH; oh++) {
                    for (int ow = 0; ow < outputW; ow++) {
                        long hStart = (long) oh * height / outputH;
                        long hEnd = (long) (oh + 1) * height / outputH;
                        long wStart = (long) ow * width / outputW;
                        long wEnd = (long) (ow + 1) * width / outputW;
                        double sum = 0d;
                        long count = 0;
                        for (long ih = hStart; ih < hEnd; ih++) {
                            for (long iw = wStart; iw < wEnd; iw++) {
                                long logicalIndex = ((n * channels + c) * height + ih) * width + iw;
                                sum += CpuTensorLayoutAccess.readLogical(source, dimensions, layout, logicalIndex);
                                count++;
                            }
                        }
                        long outputIndex = ((n * channels + c) * outputH + oh) * outputW + ow;
                        destination.set(ValueLayout.JAVA_FLOAT, outputIndex * Float.BYTES,
                                count == 0 ? 0f : (float) (sum / count));
                    }
                }
            }
        }
        return outputFactory.create(outputShape, output);
    }

    private static void requireSpatialInput(Tensor input, String operation) {
        Objects.requireNonNull(input, "input");
        if (input.shape().rank() != 4) {
            throw new IllegalArgumentException(operation + " expects NCHW rank-4 input");
        }
        CpuTensorLayoutAccess.requireF32(input);
    }
}
