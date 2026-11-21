package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.output.EmbeddingGemma2VectorPostProcessor;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImageProcessor.RgbImage;

/**
 * Image-plus-text embedding entry point for an already-loaded EmbeddingGemma 2
 * multimodal encoder.
 *
 * <p>This adapter accepts decoded RGB images and tokenizer-expanded token IDs.
 * Image decoding, remote URI fetching, tokenizer policy, and prompt expansion
 * remain at the application boundary. It performs the model forward pass,
 * mask-aware mean pooling, Matryoshka truncation, and L2 normalization.</p>
 *
 * <p>This is a model-level entry point, not yet an implementation of
 * {@code EmbeddingModelRunner}: that SPI currently accepts text-only
 * {@code EmbeddingRequest} values.</p>
 */
public final class EmbeddingGemma2ImageEmbeddingRunner {
    private final EmbeddingGemma2MultimodalEncoder encoder;

    public EmbeddingGemma2ImageEmbeddingRunner(EmbeddingGemma2MultimodalEncoder encoder) {
        this.encoder = Objects.requireNonNull(encoder, "encoder must not be null");
    }

    /** Executes one image/text request and returns a normalized retrieval vector. */
    public float[] embed(Request request) {
        Objects.requireNonNull(request, "request must not be null");
        float[] pooled = encoder.encodePooled(
                request.inputIds(), request.images(), request.validTokens(), request.positionIds());
        return EmbeddingGemma2VectorPostProcessor.process(pooled, request.outputDimension());
    }

    /**
     * One prompt and its ordered decoded images. The token sequence must already
     * contain exactly one image placeholder per projected image soft token.
     */
    public record Request(
            int[] inputIds,
            List<RgbImage> images,
            boolean[] validTokens,
            int[] positionIds,
            int outputDimension) {

        public Request {
            Objects.requireNonNull(inputIds, "inputIds must not be null");
            Objects.requireNonNull(images, "images must not be null");
            if (inputIds.length == 0) throw new IllegalArgumentException("inputIds must not be empty");
            if (images.isEmpty()) throw new IllegalArgumentException("at least one image is required");
            inputIds = inputIds.clone();
            images = List.copyOf(images);
            if (validTokens != null && validTokens.length != inputIds.length) {
                throw new IllegalArgumentException("validTokens length must equal inputIds length");
            }
            validTokens = validTokens == null ? null : validTokens.clone();
            if (positionIds == null) {
                positionIds = sequentialPositions(inputIds.length);
            } else {
                if (positionIds.length != inputIds.length) {
                    throw new IllegalArgumentException("positionIds length must equal inputIds length");
                }
                positionIds = positionIds.clone();
            }
            if (!EmbeddingGemma2VectorPostProcessor.isSupportedDimension(outputDimension)) {
                throw new IllegalArgumentException("unsupported output dimension: " + outputDimension);
            }
        }

        @Override public int[] inputIds() { return inputIds.clone(); }
        @Override public boolean[] validTokens() { return validTokens == null ? null : validTokens.clone(); }
        @Override public int[] positionIds() { return positionIds.clone(); }

        private static int[] sequentialPositions(int length) {
            int[] positions = new int[length];
            for (int index = 0; index < length; index++) positions[index] = index;
            return positions;
        }
    }
}
