package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Immutable adapter for the tensor outputs of the Gemma4-compatible image
 * processor used by EmbeddingGemma 2.
 *
 * <p>Shapes follow the processor contract:
 * {@code pixelValues[images][paddedPatches][patchPixels]},
 * {@code imagePositionIds[images][paddedPatches][2]}, and one actual
 * soft-token count per image. Pixel values are RGB patch vectors already
 * rescaled to [0,1]; patch ordering is row-major with channels-last values
 * within each patch. Padding positions are exactly (-1,-1).</p>
 *
 * <p>Resizing and RGB decoding are intentionally outside this adapter. It
 * accepts processor-produced patch tensors without introducing an image
 * codec or resampling dependency into the model module.</p>
 */
public final class EmbeddingGemma2VisionProcessorOutput {
    private final float[][][] pixelValues;
    private final int[][][] imagePositionIds;
    private final int[] numSoftTokensPerImage;

    public EmbeddingGemma2VisionProcessorOutput(
            float[][][] pixelValues,
            int[][][] imagePositionIds,
            int[] numSoftTokensPerImage) {
        Objects.requireNonNull(pixelValues, "pixelValues must not be null");
        Objects.requireNonNull(imagePositionIds, "imagePositionIds must not be null");
        Objects.requireNonNull(numSoftTokensPerImage, "numSoftTokensPerImage must not be null");
        if (pixelValues.length == 0 || pixelValues.length != imagePositionIds.length
                || pixelValues.length != numSoftTokensPerImage.length) {
            throw new IllegalArgumentException("pixel values, position IDs, and soft-token counts must have equal non-zero image count");
        }
        int patchCount = requireNonEmpty(pixelValues[0], "pixelValues[0]");
        int patchWidth = requireNonEmpty(pixelValues[0][0], "pixelValues[0][0]").length;
        if (patchWidth == 0) throw new IllegalArgumentException("patch pixel width must be positive");
        this.pixelValues = new float[pixelValues.length][patchCount][patchWidth];
        this.imagePositionIds = new int[imagePositionIds.length][patchCount][2];
        for (int image = 0; image < pixelValues.length; image++) {
            if (pixelValues[image] == null || pixelValues[image].length != patchCount
                    || imagePositionIds[image] == null || imagePositionIds[image].length != patchCount) {
                throw new IllegalArgumentException("processor outputs must use rectangular image and patch dimensions");
            }
            if (numSoftTokensPerImage[image] <= 0) {
                throw new IllegalArgumentException("numSoftTokensPerImage[" + image + "] must be positive");
            }
            boolean hasValid = false;
            for (int patch = 0; patch < patchCount; patch++) {
                float[] pixels = pixelValues[image][patch];
                int[] position = imagePositionIds[image][patch];
                if (pixels == null || pixels.length != patchWidth) {
                    throw new IllegalArgumentException("pixel patch " + patch + " for image " + image
                            + " must have width " + patchWidth);
                }
                for (float value : pixels) {
                    if (!Float.isFinite(value) || value < 0f || value > 1f) {
                        throw new IllegalArgumentException("processor pixel values must be finite and in [0,1]");
                    }
                }
                if (position == null || position.length != 2) {
                    throw new IllegalArgumentException("position " + patch + " for image " + image
                            + " must contain (x,y)");
                }
                boolean padding = position[0] == -1 && position[1] == -1;
                if (!padding && (position[0] < 0 || position[1] < 0)) {
                    throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
                }
                hasValid |= !padding;
                this.pixelValues[image][patch] = pixels.clone();
                this.imagePositionIds[image][patch] = position.clone();
            }
            if (!hasValid) throw new IllegalArgumentException("image " + image + " has no valid patches");
        }
        this.numSoftTokensPerImage = numSoftTokensPerImage.clone();
    }

    public int imageCount() { return pixelValues.length; }
    public int paddedPatchCount() { return pixelValues[0].length; }
    public int patchPixelWidth() { return pixelValues[0][0].length; }
    public float[][][] pixelValues() { return copy(pixelValues); }
    public int[][][] imagePositionIds() { return copy(imagePositionIds); }
    public int[] numSoftTokensPerImage() { return numSoftTokensPerImage.clone(); }

    private static int requireNonEmpty(float[][] values, String name) {
        if (values == null || values.length == 0) throw new IllegalArgumentException(name + " must be non-empty");
        return values.length;
    }

    private static float[] requireNonEmpty(float[] values, String name) {
        if (values == null || values.length == 0) throw new IllegalArgumentException(name + " must be non-empty");
        return values;
    }

    private static float[][][] copy(float[][][] source) {
        float[][][] result = new float[source.length][][];
        for (int i = 0; i < source.length; i++) {
            result[i] = new float[source[i].length][];
            for (int j = 0; j < source[i].length; j++) result[i][j] = source[i][j].clone();
        }
        return result;
    }

    private static int[][][] copy(int[][][] source) {
        int[][][] result = new int[source.length][][];
        for (int i = 0; i < source.length; i++) {
            result[i] = new int[source[i].length][];
            for (int j = 0; j < source[i].length; j++) result[i][j] = source[i][j].clone();
        }
        return result;
    }
}
