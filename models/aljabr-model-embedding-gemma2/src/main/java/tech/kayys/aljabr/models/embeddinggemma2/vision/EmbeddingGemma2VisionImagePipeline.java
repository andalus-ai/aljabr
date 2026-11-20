package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;

/**
 * Checkpoint-backed image-to-text-soft-token pipeline for EmbeddingGemma 2.
 *
 * <p>Composes the model-family checkpoint mappers, Gemma4-compatible image
 * processor, executable vision tower, and modality-to-text bridge. A format
 * reader supplies the named dense checkpoint tensors; this class is agnostic
 * to Safetensors, GGUF, or another storage format.</p>
 *
 * <p>Image decoding remains an application concern. Inputs are already-decoded
 * RGB images; outputs retain one text-hidden soft-token sequence per image.</p>
 */
public final class EmbeddingGemma2VisionImagePipeline {
    private final EmbeddingGemma2VisionImageProcessor imageProcessor;
    private final EmbeddingGemma2VisionTower visionTower;

    public EmbeddingGemma2VisionImagePipeline(
            EmbeddingGemma2VisionImageProcessor imageProcessor,
            EmbeddingGemma2VisionTower visionTower) {
        this.imageProcessor = Objects.requireNonNull(imageProcessor, "imageProcessor must not be null");
        this.visionTower = Objects.requireNonNull(visionTower, "visionTower must not be null");
        if (imageProcessor.patchSize() <= 0
                || imageProcessor.poolingKernelSize() != visionTower.poolingKernelSize()) {
            throw new IllegalArgumentException("image processor and vision tower geometry must agree");
        }
    }
    }

    /**
     * Builds the complete image path from named checkpoint tensors.
     *
     * @param visionSpec checkpoint-derived Gemma4 vision dimensions
     * @param tensors named dense checkpoint tensors after format decoding
     * @param textHiddenSize output width of the modality-to-text projection
     * @param visionEpsilon RMSNorm epsilon used by the vision encoder
     * @param bridgeEpsilon RMSNorm epsilon used by the modality-to-text bridge
     * @param ropeTheta axial vision RoPE theta
     * @param maxSoftTokens processor soft-token budget from image processor config
     * @param poolingKernelSize spatial pooling kernel from vision processor config
     */
    public static EmbeddingGemma2VisionImagePipeline fromCheckpoint(
            EmbeddingGemma2VisionCheckpointMapper.VisionSpec visionSpec,
            Map<String, CheckpointTensor> tensors,
            int textHiddenSize,
            float visionEpsilon,
            float bridgeEpsilon,
            double ropeTheta,
            int maxSoftTokens,
            int poolingKernelSize) {
        Objects.requireNonNull(visionSpec, "visionSpec must not be null");
        Objects.requireNonNull(tensors, "tensors must not be null");
        EmbeddingGemma2VisionCheckpointMapper.Weights visionWeights =
                EmbeddingGemma2VisionCheckpointMapper.map(visionSpec, tensors);
        EmbeddingGemma2MultimodalEmbedder bridge =
                EmbeddingGemma2MultimodalCheckpointMapper.mapVisionEmbedder(
                        visionSpec.hiddenSize(), textHiddenSize, bridgeEpsilon, tensors);
        EmbeddingGemma2VisionTower tower = EmbeddingGemma2VisionTower.fromCheckpoint(
                visionSpec, visionWeights, bridge, visionEpsilon, ropeTheta, poolingKernelSize);
        EmbeddingGemma2VisionImageProcessor processor = new EmbeddingGemma2VisionImageProcessor(
                visionSpec.patchSize(), maxSoftTokens, poolingKernelSize);
        return new EmbeddingGemma2VisionImagePipeline(processor, tower);
    }

    /** Converts one decoded RGB image into projected text-space image tokens. */
    public float[][] encode(EmbeddingGemma2VisionImageProcessor.RgbImage image) {
        return encodeBatch(List.of(Objects.requireNonNull(image, "image must not be null"))).get(0);
    }

    /**
     * Converts a batch of decoded RGB images into separate projected token
     * sequences. Image boundaries and each image's actual soft-token count are
     * preserved; padded tokens are removed by the vision tower.
     */
    public List<float[][]> encodeBatch(List<EmbeddingGemma2VisionImageProcessor.RgbImage> images) {
        EmbeddingGemma2VisionProcessorOutput processorOutput = imageProcessor.process(images);
        return visionTower.encodeAndProject(processorOutput);
    }

    /** Exposes preprocessing separately for callers that need model inputs or diagnostics. */
    public EmbeddingGemma2VisionProcessorOutput preprocess(
            List<EmbeddingGemma2VisionImageProcessor.RgbImage> images) {
        return imageProcessor.process(images);
    }

    public EmbeddingGemma2VisionImageProcessor imageProcessor() {
        return imageProcessor;
    }

}
