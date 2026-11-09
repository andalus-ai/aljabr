package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.BatchNorm2d;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.MaxPool2d;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.Sequential;

/**
 * ResNet image classification models — equivalent to {@code torchvision.models.resnet*}.
 *
 * <p>Implements the residual network architecture from
 * <em>"Deep Residual Learning for Image Recognition"</em> (He et al., 2015).
 */
public final class ResNet {

    private ResNet() {}

    /** @deprecated Use {@link #resnet18(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule resnet18(int numClasses) {
        return resnet18(null, numClasses);
    }

    public static NNModule resnet18(ComputeBackend backend, int numClasses) {
        return new ResNetModel(backend, new int[]{2, 2, 2, 2}, numClasses, false);
    }

    /** @deprecated Use {@link #resnet34(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule resnet34(int numClasses) {
        return resnet34(null, numClasses);
    }

    public static NNModule resnet34(ComputeBackend backend, int numClasses) {
        return new ResNetModel(backend, new int[]{3, 4, 6, 3}, numClasses, false);
    }

    /** @deprecated Use {@link #resnet50(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule resnet50(int numClasses) {
        return resnet50(null, numClasses);
    }

    public static NNModule resnet50(ComputeBackend backend, int numClasses) {
        return new ResNetModel(backend, new int[]{3, 4, 6, 3}, numClasses, true);
    }

    private static final class ResNetModel extends NNModule {

        private final Conv2d stem;
        private final BatchNorm2d stemBn;
        private final MaxPool2d stemPool;
        private final NNModule[] layers;
        private final Linear fc;

        ResNetModel(ComputeBackend backend, int[] layerCounts, int numClasses, boolean bottleneck) {
            super(backend);
            int[] channels = bottleneck
                    ? new int[]{256, 512, 1024, 2048}
                    : new int[]{64, 128, 256, 512};

            this.stem = register("stem", new Conv2d(backend, 3, 64, 7, 2, 3));
            this.stemBn = register("stem_bn", new BatchNorm2d(backend, 64));
            this.stemPool = register("stem_pool", new MaxPool2d(backend, 3, 2, 1));

            this.layers = new NNModule[4];
            int inChannels = 64;
            for (int i = 0; i < 4; i++) {
                layers[i] = register("layer" + (i + 1),
                        makeLayer(backend, inChannels, channels[i], layerCounts[i],
                                i > 0 ? 2 : 1, bottleneck));
                inChannels = channels[i];
            }
            this.fc = register("fc", new Linear(backend, channels[3], numClasses));
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = stemPool.forward(stemBn.forward(stem.forward(input)).relu());

            for (NNModule layer : layers) {
                output = layer.forward(output);
            }

            long[] shape = output.shape().dims();
            int batch = Math.toIntExact(shape[0]);
            int channels = Math.toIntExact(shape[1]);
            int height = Math.toIntExact(shape[2]);
            int width = Math.toIntExact(shape[3]);
            float[] pooled = new float[Math.multiplyExact(batch, channels)];
            float[] values = output.toFloatArray();
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < channels; c++) {
                    float sum = 0;
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            sum += values[b * channels * height * width
                                    + c * height * width + h * width + w];
                        }
                    }
                    pooled[b * channels + c] = sum / (height * width);
                }
            }
            return fc.forward(tensorOf(pooled, batch, channels));
        }

        private NNModule makeLayer(
                ComputeBackend backend, int inChannels, int outChannels,
                int count, int stride, boolean bottleneck) {
            return new Sequential(backend,
                    buildBlocks(backend, inChannels, outChannels, count, stride, bottleneck));
        }

        private NNModule[] buildBlocks(
                ComputeBackend backend, int inChannels, int outChannels,
                int count, int stride, boolean bottleneck) {
            NNModule[] blocks = new NNModule[count];
            blocks[0] = bottleneck
                    ? new BottleneckBlock(backend, inChannels, outChannels, stride)
                    : new BasicBlock(backend, inChannels, outChannels, stride);
            for (int i = 1; i < count; i++) {
                blocks[i] = bottleneck
                        ? new BottleneckBlock(backend, outChannels, outChannels, 1)
                        : new BasicBlock(backend, outChannels, outChannels, 1);
            }
            return blocks;
        }
    }

    static final class BottleneckBlock extends NNModule {

        private final Conv2d conv1;
        private final Conv2d conv2;
        private final Conv2d conv3;
        private final BatchNorm2d bn1;
        private final BatchNorm2d bn2;
        private final BatchNorm2d bn3;
        private final Conv2d downsample;

        BottleneckBlock(ComputeBackend backend, int inChannels, int outChannels, int stride) {
            super(backend);
            int midChannels = outChannels / 4;
            this.conv1 = register("conv1", new Conv2d(backend, inChannels, midChannels, 1));
            this.bn1 = register("bn1", new BatchNorm2d(backend, midChannels));
            this.conv2 = register("conv2", new Conv2d(backend, midChannels, midChannels, 3, stride, 1));
            this.bn2 = register("bn2", new BatchNorm2d(backend, midChannels));
            this.conv3 = register("conv3", new Conv2d(backend, midChannels, outChannels, 1));
            this.bn3 = register("bn3", new BatchNorm2d(backend, outChannels));
            this.downsample = inChannels != outChannels || stride != 1
                    ? register("ds", new Conv2d(backend, inChannels, outChannels, 1, stride, 0))
                    : null;
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor identity = input;
            Tensor output = bn1.forward(conv1.forward(input)).relu();
            output = bn2.forward(conv2.forward(output)).relu();
            output = bn3.forward(conv3.forward(output));
            if (downsample != null) {
                identity = downsample.forward(input);
            }
            return output.add(identity).relu();
        }
    }

    static final class BasicBlock extends NNModule {

        private final Conv2d conv1;
        private final Conv2d conv2;
        private final BatchNorm2d bn1;
        private final BatchNorm2d bn2;
        private final Conv2d downsample;

        BasicBlock(ComputeBackend backend, int inChannels, int outChannels, int stride) {
            super(backend);
            this.conv1 = register("conv1",
                    new Conv2d(backend, inChannels, outChannels, 3, stride, 1));
            this.bn1 = register("bn1", new BatchNorm2d(backend, outChannels));
            this.conv2 = register("conv2",
                    new Conv2d(backend, outChannels, outChannels, 3, 1, 1));
            this.bn2 = register("bn2", new BatchNorm2d(backend, outChannels));
            this.downsample = inChannels != outChannels || stride != 1
                    ? register("ds", new Conv2d(backend, inChannels, outChannels, 1, stride, 0))
                    : null;
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor identity = input;
            Tensor output = bn1.forward(conv1.forward(input)).relu();
            output = bn2.forward(conv2.forward(output));
            if (downsample != null) {
                identity = downsample.forward(input);
            }
            return output.add(identity).relu();
        }
    }
}
