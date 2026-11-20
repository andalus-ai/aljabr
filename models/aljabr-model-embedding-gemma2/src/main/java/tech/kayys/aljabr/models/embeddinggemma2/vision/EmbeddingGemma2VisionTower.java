package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;

/**
 * Composed EmbeddingGemma 2 Gemma4 vision tower.
 *
 * <p>Executes patch projection and 2D position lookup, the checkpoint-ordered
 * bidirectional vision encoder, spatial pooling, optional float32 output
 * standardization, and (when requested) the modality-to-text projection.
 * Image decoding, resizing, RGB conversion, and patch extraction remain
 * processor responsibilities.</p>
 *
 * <p>The tower removes padded pooled tokens before standardization, matching
 * the Hugging Face Gemma4VisionModel forward path. The multimodal embedder is
 * kept as a separate final operation because the Hugging Face
 * EmbeddingGemma2Model applies it after the vision tower returns.</p>
 */
public final class EmbeddingGemma2VisionTower {
    private final EmbeddingGemma2VisionEncoder encoder;
    private final EmbeddingGemma2VisionPooler pooler;
    private final EmbeddingGemma2VisionSoftTokenStandardizer standardizer;
    private final EmbeddingGemma2MultimodalEmbedder multimodalEmbedder;
    private final int poolingKernelSize;

    public EmbeddingGemma2VisionTower(
            EmbeddingGemma2VisionEncoder encoder,
            EmbeddingGemma2VisionPooler pooler,
            EmbeddingGemma2VisionSoftTokenStandardizer standardizer,
            EmbeddingGemma2MultimodalEmbedder multimodalEmbedder,
            int poolingKernelSize) {
        this.encoder = Objects.requireNonNull(encoder, "encoder must not be null");
        this.pooler = Objects.requireNonNull(pooler, "pooler must not be null");
        if (poolingKernelSize <= 0) {
            throw new IllegalArgumentException("poolingKernelSize must be positive");
        }
        this.poolingKernelSize = poolingKernelSize;
        this.standardizer = standardizer;
        this.multimodalEmbedder = multimodalEmbedder;
    }

    /**
     * Constructs the executable reference tower from mapped vision weights.
     *
     * @param spec checkpoint-derived dimensions and standardization flag
     * @param weights mapped vision checkpoint weights
     * @param epsilon Gemma RMSNorm epsilon
     * @param ropeTheta axial vision RoPE theta (Gemma4 default: 100)
     * @param poolingKernelSize configured spatial pooling kernel
     */
    public static EmbeddingGemma2VisionTower fromCheckpoint(
            EmbeddingGemma2VisionCheckpointMapper.VisionSpec spec,
            EmbeddingGemma2VisionCheckpointMapper.Weights weights,
            float epsilon,
            double ropeTheta,
            int poolingKernelSize) {
        return fromCheckpoint(spec, weights, null, epsilon, ropeTheta, poolingKernelSize);
    }

    /**
     * Constructs the complete image path, including the checkpoint-backed
     * modality-to-text bridge. The supplied embedder is normally created by
     * {@link EmbeddingGemma2MultimodalCheckpointMapper#mapVisionEmbedder}.
     */
    public static EmbeddingGemma2VisionTower fromCheckpoint(
            EmbeddingGemma2VisionCheckpointMapper.VisionSpec spec,
            EmbeddingGemma2VisionCheckpointMapper.Weights weights,
            EmbeddingGemma2MultimodalEmbedder multimodalEmbedder,
            float epsilon,
            double ropeTheta,
            int poolingKernelSize) {
        Objects.requireNonNull(spec, "spec must not be null");
        Objects.requireNonNull(weights, "weights must not be null");
        if (weights.layers().size() != spec.numHiddenLayers()) {
            throw new IllegalArgumentException("mapped vision layer count does not match vision configuration");
        }
        EmbeddingGemma2VisionPatchEmbedder patchEmbedder =
                new EmbeddingGemma2VisionPatchEmbedder(
                        spec.patchSize(), spec.hiddenSize(), spec.positionEmbeddingSize(),
                        weights.patchEmbedder());

        List<EmbeddingGemma2VisionEncoderLayer> layers =
                new ArrayList<>(spec.numHiddenLayers());
        for (int index = 0; index < spec.numHiddenLayers(); index++) {
            EmbeddingGemma2VisionCheckpointMapper.LayerWeights layerWeights = weights.layers().get(index);
            EmbeddingGemma2VisionAttention attention = new EmbeddingGemma2VisionAttention(
                    spec.hiddenSize(), spec.numAttentionHeads(), spec.numKeyValueHeads(),
                    spec.headDim(), epsilon, ropeTheta, layerWeights.attention());
            EmbeddingGemma2VisionEncoderLayer layer = new EmbeddingGemma2VisionEncoderLayer(
                    spec.hiddenSize(), spec.intermediateSize(), epsilon, attention::compute,
                    layerWeights.mlp(), layerWeights.norms());
            layers.add(layer);
        }

        EmbeddingGemma2VisionEncoder encoder = new EmbeddingGemma2VisionEncoder(
                patchEmbedder, layers, spec.numHiddenLayers());
        EmbeddingGemma2VisionPooler pooler = new EmbeddingGemma2VisionPooler(spec.hiddenSize());
        EmbeddingGemma2VisionSoftTokenStandardizer standardizer = null;
        if (spec.standardize()) {
            if (weights.standardizationBias() == null || weights.standardizationScale() == null) {
                throw new IllegalArgumentException(
                        "standardization is enabled but checkpoint standardization vectors are missing");
            }
            standardizer = new EmbeddingGemma2VisionSoftTokenStandardizer(
                    spec.hiddenSize(), weights.standardizationBias(), weights.standardizationScale());
        }
        return new EmbeddingGemma2VisionTower(
                encoder, pooler, standardizer, multimodalEmbedder, poolingKernelSize);
    }

    /**
     * Runs one image through the vision tower and returns valid pooled tokens.
     *
     * @param patchPixels RGB patch vectors [patches][3*patchSize*patchSize]
     * @param pixelPositionIds (x,y) coordinates; (-1,-1) denotes padding
     * @param outputLength desired spatial soft-token count before padding removal
     */
    public VisionOutput encode(float[][] patchPixels, int[][] pixelPositionIds, int outputLength) {
        Objects.requireNonNull(patchPixels, "patchPixels must not be null");
        Objects.requireNonNull(pixelPositionIds, "pixelPositionIds must not be null");
        if (patchPixels.length != pixelPositionIds.length || patchPixels.length == 0) {
            throw new IllegalArgumentException("patch pixels and position IDs must have the same non-zero length");
        }
        int poolingArea = Math.multiplyExact(poolingKernelSize, poolingKernelSize);
        if (Math.multiplyExact(outputLength, poolingArea) != patchPixels.length) {
            throw new IllegalArgumentException("patch count must equal outputLength * poolingKernelSize^2");
        }
        boolean[] validPatches = validMask(pixelPositionIds);
        float[][] encoded = encoder.encodePatches(patchPixels, pixelPositionIds, validPatches);
        EmbeddingGemma2VisionPooler.Result pooled =
                pooler.pool(encoded, pixelPositionIds, invert(validPatches), outputLength);
        float[][] validStates = removePadding(pooled.hiddenStates(), pooled.paddingMask());
        if (standardizer != null) {
            validStates = standardizer.standardize(validStates);
        }
        return new VisionOutput(validStates, poolingKernelSize);
    }

    /**
     * Runs the tower and projects its valid soft tokens into text hidden space.
     * A multimodal embedder may be supplied at construction; otherwise it must
     * be passed explicitly here.
     */
    public float[][] encodeAndProject(
            float[][] patchPixels,
            int[][] pixelPositionIds,
            int outputLength,
            EmbeddingGemma2MultimodalEmbedder embedder) {
        EmbeddingGemma2MultimodalEmbedder projector =
                embedder != null ? embedder : multimodalEmbedder;
        if (projector == null) {
            throw new IllegalStateException("no multimodal-to-text embedder was configured");
        }
        return projector.project(encode(patchPixels, pixelPositionIds, outputLength).hiddenStates());
    }

    public int poolingKernelSize() {
        return poolingKernelSize;
    }

    private static boolean[] validMask(int[][] positions) {
        boolean[] valid = new boolean[positions.length];
        for (int i = 0; i < positions.length; i++) {
            if (positions[i] == null || positions[i].length != 2) {
                throw new IllegalArgumentException("pixel position " + i + " must contain (x,y)");
            }
            int x = positions[i][0], y = positions[i][1];
            if (!((x == -1 && y == -1) || (x >= 0 && y >= 0))) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
            valid[i] = x >= 0;
        }
        return valid;
    }

    private static boolean[] invert(boolean[] valid) {
        boolean[] padding = new boolean[valid.length];
        for (int i = 0; i < valid.length; i++) padding[i] = !valid[i];
        return padding;
    }

    private static float[][] removePadding(float[][] states, boolean[] padding) {
        int count = 0;
        for (boolean isPadding : padding) if (!isPadding) count++;
        if (count == 0) {
            throw new IllegalArgumentException("vision pooling produced no valid soft tokens");
        }
        float[][] result = new float[count][];
        int output = 0;
        for (int i = 0; i < states.length; i++) {
            if (!padding[i]) result[output++] = states[i].clone();
        }
        return result;
    }

    /** Valid pooled vision tokens, with padding already removed. */
    public static final class VisionOutput {
        private final float[][] hiddenStates;
        private final int poolingKernelSize;

        private VisionOutput(float[][] hiddenStates, int poolingKernelSize) {
            this.hiddenStates = deepCopy(hiddenStates);
            this.poolingKernelSize = poolingKernelSize;
        }

        public float[][] hiddenStates() {
            return deepCopy(hiddenStates);
        }

        public int poolingKernelSize() {
            return poolingKernelSize;
        }

        private static float[][] deepCopy(float[][] values) {
            float[][] copy = new float[values.length][];
            for (int i = 0; i < values.length; i++) copy[i] = values[i].clone();
            return copy;
        }
    }
}
