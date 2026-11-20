package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.output.EmbeddingGemma2MeanPooler;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImagePipeline;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImageProcessor.RgbImage;

/**
 * End-to-end image-and-text encoder for EmbeddingGemma 2.
 *
 * <p>This model-level composition executes decoded RGB images through the
 * checkpoint-backed vision pipeline, replaces the corresponding image
 * placeholder embeddings in the text sequence, and runs the assembled
 * sequence through the complete text backbone. It is deliberately independent
 * of tokenizer, checkpoint-file, and image-codec implementations.</p>
 *
 * <p>The caller must provide an already-expanded token sequence containing
 * exactly one image placeholder per projected image soft token, in image
 * order. Prompt expansion and tokenizer-specific special-token policy remain
 * the responsibility of the tokenizer/prompt layer.</p>
 */
public final class EmbeddingGemma2MultimodalEncoder {
    private final EmbeddingGemma2TextBackbone textBackbone;
    private final EmbeddingGemma2VisionImagePipeline visionPipeline;
    private final int imageTokenId;
    private final int padTokenId;

    public EmbeddingGemma2MultimodalEncoder(
            EmbeddingGemma2TextBackbone textBackbone,
            EmbeddingGemma2VisionImagePipeline visionPipeline,
            int imageTokenId,
            int padTokenId) {
        this.textBackbone = Objects.requireNonNull(textBackbone, "textBackbone must not be null");
        this.visionPipeline = Objects.requireNonNull(visionPipeline, "visionPipeline must not be null");
        if (imageTokenId < 0) {
            throw new IllegalArgumentException("imageTokenId must be non-negative");
        }
        if (padTokenId < 0) {
            throw new IllegalArgumentException("padTokenId must be non-negative");
        }
        this.imageTokenId = imageTokenId;
        this.padTokenId = padTokenId;
    }

    /**
     * Encodes one prompt with its decoded images into projected token states.
     *
     * @param inputIds expanded text token IDs, including image placeholders
     * @param images decoded RGB images in the same order as their prompt slots
     * @param validTokens optional true=non-padding mask; null means all valid
     * @param positionIds explicit position IDs for the expanded text sequence
     * @return learned per-token projected states [sequence][embeddingDim]
     */
    public float[][] encode(
            int[] inputIds,
            List<RgbImage> images,
            boolean[] validTokens,
            int[] positionIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        Objects.requireNonNull(images, "images must not be null");
        if (images.isEmpty()) {
            throw new IllegalArgumentException("at least one image is required");
        }
        List<float[][]> imageFeatures = visionPipeline.encodeBatch(images);
        float[][] mergedEmbeddings = EmbeddingGemma2MultimodalInputAssembler.assemble(
                textBackbone, inputIds, imageTokenId, padTokenId, imageFeatures);
        return textBackbone.encodeEmbeddings(mergedEmbeddings, validTokens, positionIds);
    }

    /**
     * Encodes and mask-aware mean-pools a multimodal prompt into one vector.
     * Matryoshka truncation and L2 normalization remain a separate post-process.
     */
    public float[] encodePooled(
            int[] inputIds,
            List<RgbImage> images,
            boolean[] validTokens,
            int[] positionIds) {
        float[][] tokenStates = encode(inputIds, images, validTokens, positionIds);
        boolean[] poolingMask = validTokens == null ? allValid(inputIds.length) : validTokens;
        return EmbeddingGemma2MeanPooler.meanPool(tokenStates, poolingMask);
    }

    private static boolean[] allValid(int length) {
        boolean[] mask = new boolean[length];
        java.util.Arrays.fill(mask, true);
        return mask;
    }
}
