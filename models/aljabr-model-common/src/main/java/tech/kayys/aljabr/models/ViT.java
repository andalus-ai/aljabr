package tech.kayys.aljabr.models;

import java.util.List;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorOps;
import tech.kayys.aljabr.core.nn.LayerNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.PositionalEncoding;
import tech.kayys.aljabr.core.nn.TransformerBlock;

/** Vision Transformer family for image classification. */
public final class ViT {

    private ViT() {}

    /** @deprecated Use {@link #vitBase(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule vitBase(int numClasses) {
        return vitBase(null, numClasses);
    }

    public static NNModule vitBase(ComputeBackend backend, int numClasses) {
        return new ViTModel(backend, 224, 16, 3, 768, 12, 12, 3072, numClasses, 0.1f);
    }

    /** @deprecated Use {@link #vitSmall(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule vitSmall(int numClasses) {
        return vitSmall(null, numClasses);
    }

    public static NNModule vitSmall(ComputeBackend backend, int numClasses) {
        return new ViTModel(backend, 224, 16, 3, 384, 6, 12, 1536, numClasses, 0.1f);
    }

    /** @deprecated Use {@link #vitTiny(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule vitTiny(int numClasses) {
        return vitTiny(null, numClasses);
    }

    public static NNModule vitTiny(ComputeBackend backend, int numClasses) {
        return new ViTModel(backend, 224, 16, 3, 192, 3, 12, 768, numClasses, 0.0f);
    }

    static final class ViTModel extends NNModule {

        private final int patchSize;
        private final int numPatches;
        private final int dModel;
        private final Linear patchEmbed;
        private final Tensor clsToken;
        private final PositionalEncoding posEnc;
        private final TransformerBlock[] blocks;
        private final LayerNorm norm;
        private final Linear head;

        ViTModel(ComputeBackend backend, int imgSize, int patchSize, int inChannels,
                 int dModel, int nHeads, int nLayers, int dFF, int numClasses, float dropout) {
            super(backend);
            if (imgSize <= 0 || patchSize <= 0 || imgSize % patchSize != 0
                    || inChannels <= 0 || dModel <= 0 || nHeads <= 0
                    || nLayers <= 0 || dFF <= 0 || numClasses <= 0) {
                throw new IllegalArgumentException("Invalid ViT configuration");
            }
            this.patchSize = patchSize;
            this.dModel = dModel;
            int gridSize = imgSize / patchSize;
            this.numPatches = Math.multiplyExact(gridSize, gridSize);

            this.patchEmbed = register("patch_embed",
                    new Linear(backend, inChannels * patchSize * patchSize, dModel));
            this.clsToken = register("cls_token", randomNormal(1, 1, dModel));
            this.posEnc = register("pos_enc",
                    new PositionalEncoding(backend, dModel, numPatches + 1));
            this.blocks = new TransformerBlock[nLayers];
            for (int i = 0; i < nLayers; i++) {
                blocks[i] = register("block_" + i,
                        new TransformerBlock(backend, dModel, nHeads, dFF, dropout));
            }
            this.norm = register("norm", new LayerNorm(backend, dModel));
            this.head = register("head", new Linear(backend, dModel, numClasses));
        }

        @Override
        public Tensor forward(Tensor input) {
            validateInputBackend(input);
            long[] shape = input.shape().dims();
            if (shape.length != 4) {
                throw new IllegalArgumentException("ViT expects image input [B,C,H,W]");
            }
            int batch = Math.toIntExact(shape[0]);
            int channels = Math.toIntExact(shape[1]);
            int height = Math.toIntExact(shape[2]);
            int width = Math.toIntExact(shape[3]);
            if (height % patchSize != 0 || width % patchSize != 0) {
                throw new IllegalArgumentException("Image height and width must be divisible by patch size");
            }

            int gridH = height / patchSize;
            int gridW = width / patchSize;
            int patchesPerImage = Math.multiplyExact(gridH, gridW);
            int patchFeatures = Math.multiplyExact(channels,
                    Math.multiplyExact(patchSize, patchSize));
            float[] patches = extractPatches(input.toFloatArray(), batch, channels,
                    height, width, patchSize, gridH, gridW);
            Tensor tokens = patchEmbed.forward(
                    tensorOf(patches, batch, patchesPerImage, patchFeatures));

            float[] clsValues = clsToken.toFloatArray();
            float[] batchedCls = new float[Math.multiplyExact(batch, dModel)];
            for (int b = 0; b < batch; b++) {
                System.arraycopy(clsValues, 0, batchedCls, b * dModel, dModel);
            }
            Tensor cls = tensorOf(batchedCls, batch, 1, dModel);
            tokens = TensorOps.cat(List.of(cls, tokens), 1);

            tokens = posEnc.forward(tokens);
            for (TransformerBlock block : blocks) {
                tokens = block.forward(tokens);
            }
            tokens = norm.forward(tokens);

            float[] tokenValues = tokens.toFloatArray();
            float[] clsOutput = new float[Math.multiplyExact(batch, dModel)];
            for (int b = 0; b < batch; b++) {
                System.arraycopy(tokenValues, b * (patchesPerImage + 1) * dModel,
                        clsOutput, b * dModel, dModel);
            }
            return head.forward(tensorOf(clsOutput, batch, dModel));
        }

        private float[] extractPatches(float[] image, int batch, int channels,
                                       int height, int width, int patch,
                                       int gridH, int gridW) {
            int patchCount = Math.multiplyExact(gridH, gridW);
            int patchFeatures = Math.multiplyExact(channels, Math.multiplyExact(patch, patch));
            float[] output = new float[Math.multiplyExact(batch,
                    Math.multiplyExact(patchCount, patchFeatures))];
            for (int b = 0; b < batch; b++) {
                for (int gh = 0; gh < gridH; gh++) {
                    for (int gw = 0; gw < gridW; gw++) {
                        int patchIndex = b * patchCount + gh * gridW + gw;
                        int base = patchIndex * patchFeatures;
                        for (int c = 0; c < channels; c++) {
                            for (int ph = 0; ph < patch; ph++) {
                                for (int pw = 0; pw < patch; pw++) {
                                    int imageIndex = b * channels * height * width
                                            + c * height * width
                                            + (gh * patch + ph) * width + gw * patch + pw;
                                    output[base + c * patch * patch + ph * patch + pw]
                                            = image[imageIndex];
                                }
                            }
                        }
                    }
                }
            }
            return output;
        }
    }
}
