package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Immutable adapter for the tensor outputs of the Gemma4-compatible image
 * processor used by EmbeddingGemma 2.
 *
 * <p>Shapes follow the processor contract:
 * {@code pixelValues[images][paddedPatches][patchPixels]},
 * {@code imagePositionIds[images][paddedPatches][2]}, and one actual
 * soft-token count per image. Pixel values are RGB patch vectors rescaled by
 * 1/255; bicubic antialias filtering can produce small values outside [0,1],
 * so finite overshoot is retained exactly as in torchvision. Patch ordering
 * is row-major with channels-last values within each patch. Padding positions
 * are exactly (-1,-1) and form a contiguous suffix.</p>
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
            int validPatchCount = 0;
            int maxX = -1;
            boolean paddingStarted = false;
            for (int patch = 0; patch < patchCount; patch++) {
                float[] pixels = pixelValues[image][patch];
                int[] position = imagePositionIds[image][patch];
                if (pixels == null || pixels.length != patchWidth) {
                    throw new IllegalArgumentException("pixel patch " + patch + " for image " + image
                            + " must have width " + patchWidth);
                }
                for (float value : pixels) {
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("processor pixel values must be finite");
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
                if (padding) {
                    paddingStarted = true;
                } else {
                    if (paddingStarted) {
                        throw new IllegalArgumentException("valid patch positions must precede the padded suffix");
                    }
                    validPatchCount++;
                    maxX = Math.max(maxX, position[0]);
                }
                this.pixelValues[image][patch] = pixels.clone();
                this.imagePositionIds[image][patch] = position.clone();
            }
            validateGeometry(image, patchCount, validPatchCount, maxX, numSoftTokensPerImage[image]);
        }
        this.numSoftTokensPerImage = numSoftTokensPerImage.clone();
    }

    public int imageCount() { return pixelValues.length; }
    public int paddedPatchCount() { return pixelValues[0].length; }
    public int patchPixelWidth() { return pixelValues[0][0].length; }
    public float[][][] pixelValues() { return copy(pixelValues); }
    public int[][][] imagePositionIds() { return copy(imagePositionIds); }
    public int[] numSoftTokensPerImage() { return numSoftTokensPerImage.clone(); }

    private void validateGeometry(
            int image, int paddedPatchCount, int validPatchCount, int maxX, int softTokenCount) {
        if (validPatchCount == 0) {
            throw new IllegalArgumentException("image " + image + " has no valid patches");
        }
        if (validPatchCount % softTokenCount != 0) {
            throw new IllegalArgumentException("image " + image
                    + " patch count is not divisible by its declared soft-token count");
        }
        int poolingArea = validPatchCount / softTokenCount;
        int poolingKernel = (int) Math.sqrt(poolingArea);
        if (poolingKernel <= 0 || poolingKernel * poolingKernel != poolingArea
                || paddedPatchCount % poolingArea != 0) {
            throw new IllegalArgumentException("image " + image
                    + " patch and soft-token counts do not describe a square pooling grid");
        }
        if (maxX == Integer.MAX_VALUE) {
            throw new IllegalArgumentException("image " + image + " patch grid width overflows an integer");
        }
        int gridWidth = maxX + 1;
        if (validPatchCount % gridWidth != 0) {
            throw new IllegalArgumentException("image " + image + " valid patches do not form a rectangular grid");
        }
        int gridHeight = validPatchCount / gridWidth;
        if (gridWidth % poolingKernel != 0 || gridHeight % poolingKernel != 0) {
            throw new IllegalArgumentException("image " + image
                    + " patch grid dimensions must both be divisible by the inferred pooling kernel");
        }
        for (int patch = 0; patch < validPatchCount; patch++) {
            int expectedX = patch % gridWidth;
            int expectedY = patch / gridWidth;
            int[] position = imagePositionIds[image][patch];
            if (position[0] != expectedX || position[1] != expectedY) {
                throw new IllegalArgumentException("image " + image
                        + " patch positions must form a row-major rectangular grid; invalid patch " + patch);
            }
        }
    }

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
