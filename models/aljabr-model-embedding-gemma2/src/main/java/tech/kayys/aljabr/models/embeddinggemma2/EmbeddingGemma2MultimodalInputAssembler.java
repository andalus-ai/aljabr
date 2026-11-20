package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.List;
import java.util.Objects;

/**
 * Replaces image placeholder-token embeddings with ordered image soft tokens.
 *
 * <p>EmbeddingGemma 2 keeps the text sequence length unchanged: the processor
 * expands image placeholders to the exact number of soft-token slots expected
 * from the image tower. Image features are concatenated in image order and
 * scattered into those slots in token order, matching the reference model's
 * multimodal merge. The returned matrix is ready for
 * {@link EmbeddingGemma2TextBackbone#encodeEmbeddings(float[][], boolean[], int[])}.</p>
 */
public final class EmbeddingGemma2MultimodalInputAssembler {
    private EmbeddingGemma2MultimodalInputAssembler() {
    }

    /**
     * Looks up and scales text embeddings, then replaces image placeholder
     * rows with the supplied text-space image features.
     *
     * @param textBackbone text embedding owner
     * @param inputIds token IDs [sequence]
     * @param imageTokenId configured image placeholder token ID
     * @param padTokenId in-vocabulary token used for placeholder lookup before replacement
     * @param imageFeatures one projected text-hidden sequence per image, in prompt order
     * @return merged embeddings [sequence][textHiddenSize]
     */
    public static float[][] assemble(
            EmbeddingGemma2TextBackbone textBackbone,
            int[] inputIds,
            int imageTokenId,
            int padTokenId,
            List<float[][]> imageFeatures) {
        Objects.requireNonNull(textBackbone, "textBackbone must not be null");
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        int[] lookupIds = inputIds.clone();
        for (int index = 0; index < lookupIds.length; index++) {
            if (lookupIds[index] == imageTokenId) lookupIds[index] = padTokenId;
        }
        float[][] textEmbeddings = textBackbone.embedInputTokens(lookupIds);
        return replaceImagePlaceholders(textEmbeddings, inputIds, imageTokenId, imageFeatures);
    }

    /**
     * Replaces image-token rows in an already embedded sequence. This lower
     * level operation is useful when embeddings are supplied by another text
     * embedding implementation.
     */
    public static float[][] replaceImagePlaceholders(
            float[][] textEmbeddings,
            int[] inputIds,
            int imageTokenId,
            List<float[][]> imageFeatures) {
        Objects.requireNonNull(textEmbeddings, "textEmbeddings must not be null");
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        Objects.requireNonNull(imageFeatures, "imageFeatures must not be null");
        if (imageTokenId < 0) {
            throw new IllegalArgumentException("imageTokenId must be non-negative");
        }
        if (inputIds.length == 0 || textEmbeddings.length != inputIds.length) {
            throw new IllegalArgumentException("text embeddings and input IDs must have equal non-zero length");
        }

        int hiddenSize = -1;
        float[][] output = new float[textEmbeddings.length][];
        int placeholderCount = 0;
        for (int token = 0; token < textEmbeddings.length; token++) {
            float[] row = textEmbeddings[token];
            if (row == null || row.length == 0 || (hiddenSize >= 0 && row.length != hiddenSize)) {
                throw new IllegalArgumentException("textEmbeddings must be a non-empty rectangular matrix");
            }
            hiddenSize = row.length;
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("textEmbeddings must contain only finite values");
                }
            }
            output[token] = row.clone();
            if (inputIds[token] == imageTokenId) placeholderCount++;
        }

        int featureCount = 0;
        for (int image = 0; image < imageFeatures.size(); image++) {
            float[][] features = imageFeatures.get(image);
            if (features == null || features.length == 0) {
                throw new IllegalArgumentException("image feature sequence " + image + " must be non-empty");
            }
            featureCount = Math.addExact(featureCount, features.length);
            for (int token = 0; token < features.length; token++) {
                float[] row = features[token];
                if (row == null || row.length != hiddenSize) {
                    throw new IllegalArgumentException("image feature " + image + ":" + token
                            + " must have text hidden size " + hiddenSize);
                }
                for (float value : row) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("image features must contain only finite values");
                    }
                }
            }
        }
        if (placeholderCount != featureCount) {
            throw new IllegalArgumentException("image placeholder count (" + placeholderCount
                    + ") must equal projected image soft-token count (" + featureCount + ")");
        }

        int imageIndex = 0;
        int imageTokenIndex = 0;
        for (int token = 0; token < inputIds.length; token++) {
            if (inputIds[token] != imageTokenId) continue;
            while (imageIndex < imageFeatures.size()
                    && imageTokenIndex == imageFeatures.get(imageIndex).length) {
                imageIndex++;
                imageTokenIndex = 0;
            }
            output[token] = imageFeatures.get(imageIndex)[imageTokenIndex++].clone();
        }
        return output;
    }
}
