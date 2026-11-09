package tech.kayys.aljabr.models;

import java.util.ArrayList;
import java.util.List;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.AdaptiveAvgPool2d;
import tech.kayys.aljabr.core.nn.BatchNorm2d;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Dropout;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.MaxPool2d;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.ReLU;

/** VGG network family — deep CNNs with small 3×3 convolutions. */
public final class VGG {

    private VGG() {}

    /** @deprecated Use {@link #vgg11(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule vgg11(int numClasses) {
        return vgg11(null, numClasses);
    }

    public static NNModule vgg11(ComputeBackend backend, int numClasses) {
        return new VGGModel(backend, new int[][]{
                {64}, {128}, {256, 256}, {512, 512}, {512, 512}
        }, numClasses);
    }

    /** @deprecated Use {@link #vgg16(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule vgg16(int numClasses) {
        return vgg16(null, numClasses);
    }

    public static NNModule vgg16(ComputeBackend backend, int numClasses) {
        return new VGGModel(backend, new int[][]{
                {64, 64}, {128, 128}, {256, 256, 256},
                {512, 512, 512}, {512, 512, 512}
        }, numClasses);
    }

    static final class VGGModel extends NNModule {

        private final NNModule[] features;
        private final AdaptiveAvgPool2d pool;
        private final Linear fc1;
        private final Linear fc2;
        private final Linear fc3;
        private final Dropout drop;

        VGGModel(ComputeBackend backend, int[][] blockChannels, int numClasses) {
            super(backend);
            List<NNModule> layers = new ArrayList<>();
            int inChannels = 3;
            for (int[] block : blockChannels) {
                for (int outChannels : block) {
                    layers.add(new Conv2d(backend, inChannels, outChannels, 3, 1, 1));
                    layers.add(new BatchNorm2d(backend, outChannels));
                    layers.add(new ReLU(backend));
                    inChannels = outChannels;
                }
                layers.add(new MaxPool2d(backend, 2));
            }

            this.features = new NNModule[layers.size()];
            for (int i = 0; i < layers.size(); i++) {
                features[i] = register("feat_" + i, layers.get(i));
            }
            this.pool = register("pool", new AdaptiveAvgPool2d(backend, 7, 7));
            this.fc1 = register("fc1", new Linear(backend, 512L * 7 * 7, 4096));
            this.fc2 = register("fc2", new Linear(backend, 4096, 4096));
            this.fc3 = register("fc3", new Linear(backend, 4096, numClasses));
            this.drop = register("drop", new Dropout(backend, 0.5f));
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            Tensor output = input;
            for (NNModule feature : features) {
                output = feature.forward(output);
            }
            output = pool.forward(output);
            long[] shape = output.shape().dims();
            output = output.reshape(Math.toIntExact(shape[0]),
                    Math.toIntExact(output.numel() / shape[0]));
            output = drop.forward(fc1.forward(output).relu());
            output = drop.forward(fc2.forward(output).relu());
            return fc3.forward(output);
        }
    }
}
