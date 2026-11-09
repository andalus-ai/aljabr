package tech.kayys.aljabr.models;

import java.util.ArrayList;
import java.util.List;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.AdaptiveAvgPool2d;
import tech.kayys.aljabr.core.nn.BatchNorm2d;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;

/** EfficientNet model family using MBConv-style stages. */
public final class EfficientNet {

    private EfficientNet() {}

    /** @deprecated Use {@link #efficientNetB0(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule efficientNetB0(int numClasses) {
        return efficientNetB0(null, numClasses);
    }

    public static NNModule efficientNetB0(ComputeBackend backend, int numClasses) {
        return new EfficientNetModel(backend,
                new int[]{16, 24, 40, 80, 112, 192, 320},
                new int[]{1, 2, 2, 3, 3, 4, 1}, numClasses, 1280);
    }

    /** @deprecated Use {@link #efficientNetB1(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule efficientNetB1(int numClasses) {
        return efficientNetB1(null, numClasses);
    }

    public static NNModule efficientNetB1(ComputeBackend backend, int numClasses) {
        return new EfficientNetModel(backend,
                new int[]{16, 24, 40, 80, 112, 192, 320},
                new int[]{2, 3, 3, 4, 4, 5, 2}, numClasses, 1280);
    }

    static final class EfficientNetModel extends NNModule {

        private final Conv2d stemConv;
        private final BatchNorm2d stemBn;
        private final NNModule[] blocks;
        private final Conv2d headConv;
        private final BatchNorm2d headBn;
        private final AdaptiveAvgPool2d pool;
        private final Linear classifier;

        EfficientNetModel(ComputeBackend backend, int[] channels, int[] repeats,
                          int numClasses, int headChannels) {
            super(backend);
            if (channels == null || repeats == null || channels.length == 0
                    || channels.length != repeats.length || numClasses <= 0 || headChannels <= 0) {
                throw new IllegalArgumentException("Invalid EfficientNet configuration");
            }
            this.stemConv = register("stem_conv", new Conv2d(backend, 3, 32, 3, 2, 1));
            this.stemBn = register("stem_bn", new BatchNorm2d(backend, 32));

            int inChannels = 32;
            List<NNModule> blockList = new ArrayList<>();
            for (int stage = 0; stage < channels.length; stage++) {
                if (channels[stage] <= 0 || repeats[stage] <= 0) {
                    throw new IllegalArgumentException("Stage channels and repeats must be positive");
                }
                int outChannels = channels[stage];
                for (int repeat = 0; repeat < repeats[stage]; repeat++) {
                    int stride = repeat == 0 && stage > 0 ? 2 : 1;
                    blockList.add(new MBConvBlock(backend, inChannels, outChannels, stride));
                    inChannels = outChannels;
                }
            }
            this.blocks = new NNModule[blockList.size()];
            for (int i = 0; i < blockList.size(); i++) {
                blocks[i] = register("block_" + i, blockList.get(i));
            }

            this.headConv = register("head_conv", new Conv2d(backend, inChannels, headChannels, 1));
            this.headBn = register("head_bn", new BatchNorm2d(backend, headChannels));
            this.pool = register("pool", new AdaptiveAvgPool2d(backend, 1, 1));
            this.classifier = register("classifier", new Linear(backend, headChannels, numClasses));
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = stemBn.forward(stemConv.forward(input)).relu();
            for (NNModule block : blocks) {
                output = block.forward(output);
            }
            output = headBn.forward(headConv.forward(output)).relu();
            output = pool.forward(output);
            long[] shape = output.shape().dims();
            output = output.reshape(Math.toIntExact(shape[0]), Math.toIntExact(shape[1]));
            return classifier.forward(output);
        }
    }

    /** Simplified mobile inverted bottleneck block. */
    static final class MBConvBlock extends NNModule {

        private final Conv2d expand;
        private final Conv2d depthwise;
        private final Conv2d project;
        private final BatchNorm2d bn1;
        private final BatchNorm2d bn2;
        private final BatchNorm2d bn3;
        private final boolean useResidual;

        MBConvBlock(ComputeBackend backend, int inChannels, int outChannels, int stride) {
            super(backend);
            if (inChannels <= 0 || outChannels <= 0 || stride <= 0) {
                throw new IllegalArgumentException("Channels and stride must be positive");
            }
            int expandedChannels = Math.multiplyExact(inChannels, 6);
            this.expand = register("expand", new Conv2d(backend, inChannels, expandedChannels, 1));
            this.bn1 = register("bn1", new BatchNorm2d(backend, expandedChannels));
            this.depthwise = register("dw",
                    new Conv2d(backend, expandedChannels, expandedChannels, 3, stride, 1));
            this.bn2 = register("bn2", new BatchNorm2d(backend, expandedChannels));
            this.project = register("project", new Conv2d(backend, expandedChannels, outChannels, 1));
            this.bn3 = register("bn3", new BatchNorm2d(backend, outChannels));
            this.useResidual = stride == 1 && inChannels == outChannels;
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = bn1.forward(expand.forward(input)).relu();
            output = bn2.forward(depthwise.forward(output)).relu();
            output = bn3.forward(project.forward(output));
            return useResidual ? output.add(input) : output;
        }
    }
}
