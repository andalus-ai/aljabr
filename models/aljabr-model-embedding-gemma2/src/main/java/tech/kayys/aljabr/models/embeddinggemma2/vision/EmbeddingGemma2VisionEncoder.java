package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.List;
import java.util.Objects;

/**
 * Configurable Gemma4 vision encoder stack used by EmbeddingGemma 2.
 *
 * <p>Executes the checkpoint-ordered vision encoder layers over one image's
 * patch sequence. The stack computes no extra final normalization: the
 * reference Gemma4 vision encoder returns the final layer's hidden states
 * directly. Shared positional embeddings are mathematically equivalent to
 * each layer applying the same configured axial rotary transform; the current
 * float32 attention primitive computes those tables within its attention call.</p>
 *
 * <p>Image resizing, patch extraction, patch ordering, pooling, output
 * standardization, and modality-to-text projection remain outside this
 * encoder. The optional patch embedder lets callers enter at the raw RGB-patch
 * boundary or supply already embedded patches.</p>
 */
public final class EmbeddingGemma2VisionEncoder {
    private final EmbeddingGemma2VisionPatchEmbedder patchEmbedder;
    private final List<EmbeddingGemma2VisionEncoderLayer> layers;
    private final int expectedLayerCount;

    public EmbeddingGemma2VisionEncoder(
            EmbeddingGemma2VisionPatchEmbedder patchEmbedder,
            List<EmbeddingGemma2VisionEncoderLayer> layers,
            int expectedLayerCount) {
        if (expectedLayerCount <= 0) {
            throw new IllegalArgumentException("expectedLayerCount must be positive");
        }
        Objects.requireNonNull(layers, "layers must not be null");
        if (layers.size() != expectedLayerCount) {
            throw new IllegalArgumentException("expected " + expectedLayerCount
                    + " vision encoder layers but received " + layers.size());
        }
        for (int i = 0; i < layers.size(); i++) {
            Objects.requireNonNull(layers.get(i), "vision encoder layer " + i + " must not be null");
        }
        this.patchEmbedder = patchEmbedder;
        this.layers = List.copyOf(layers);
        this.expectedLayerCount = expectedLayerCount;
    }

    /**
     * Embeds RGB patches and runs every configured vision encoder layer.
     *
     * @param patchPixels flattened RGB patch values [patches][3*patchSize*patchSize]
     * @param pixelPositionIds patch coordinates (x,y), with (-1,-1) padding
     * @param validPatches optional true=valid patch mask
     * @return final unpooled vision states [patches][hiddenSize]
     */
    public float[][] encodePatches(
            float[][] patchPixels, int[][] pixelPositionIds, boolean[] validPatches) {
        if (patchEmbedder == null) {
            throw new IllegalStateException("this encoder was constructed without a patch embedder");
        }
        float[][] embeddings = patchEmbedder.embed(patchPixels, pixelPositionIds);
        return encodeEmbeddings(embeddings, pixelPositionIds, validPatches);
    }

    /**
     * Runs the encoder over precomputed patch embeddings.
     *
     * @param patchEmbeddings input [patches][hiddenSize]
     * @param pixelPositionIds patch coordinates (x,y), with (-1,-1) padding
     * @param validPatches optional true=valid patch mask
     * @return final unpooled vision states [patches][hiddenSize]
     */
    public float[][] encodeEmbeddings(
            float[][] patchEmbeddings, int[][] pixelPositionIds, boolean[] validPatches) {
        validateInputs(patchEmbeddings, pixelPositionIds, validPatches);
        float[][] hiddenStates = copy(patchEmbeddings);
        for (EmbeddingGemma2VisionEncoderLayer layer : layers) {
            hiddenStates = layer.compute(hiddenStates, pixelPositionIds, validPatches);
        }
        return hiddenStates;
    }

    public int layerCount() {
        return layers.size();
    }

    public int expectedLayerCount() {
        return expectedLayerCount;
    }

    private static void validateInputs(float[][] embeddings, int[][] positions, boolean[] validPatches) {
        if (embeddings == null || embeddings.length == 0) {
            throw new IllegalArgumentException("patchEmbeddings must be non-empty");
        }
        if (positions == null || positions.length != embeddings.length) {
            throw new IllegalArgumentException("pixelPositionIds length must equal patch sequence length");
        }
        if (validPatches != null && validPatches.length != embeddings.length) {
            throw new IllegalArgumentException("validPatches length must equal patch sequence length");
        }
        boolean hasValidPatch = false;
        int hiddenSize = -1;
        for (int i = 0; i < embeddings.length; i++) {
            float[] row = embeddings[i];
            if (row == null || row.length == 0 || (hiddenSize >= 0 && row.length != hiddenSize)) {
                throw new IllegalArgumentException("patchEmbeddings must be a non-empty rectangular matrix");
            }
            hiddenSize = row.length;
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("patchEmbeddings must contain finite values");
                }
            }
            int[] position = positions[i];
            if (position == null || position.length != 2) {
                throw new IllegalArgumentException("pixel position " + i + " must contain (x,y)");
            }
            boolean padding = position[0] == -1 && position[1] == -1;
            if (!padding && (position[0] < 0 || position[1] < 0)) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
            if (padding && validPatches != null && validPatches[i]) {
                throw new IllegalArgumentException("a padding position cannot be marked as valid");
            }
            if (!padding && (validPatches == null || validPatches[i])) hasValidPatch = true;
        }
        if (!hasValidPatch) {
            throw new IllegalArgumentException("vision encoder requires at least one valid patch");
        }
    }

    private static float[][] copy(float[][] input) {
        float[][] output = new float[input.length][];
        for (int i = 0; i < input.length; i++) output[i] = input[i].clone();
        return output;
    }
}
