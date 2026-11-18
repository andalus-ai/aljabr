package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Splits the projected, flattened vision soft-token stream back into one
 * sequence per input image.
 *
 * <p>This follows EmbeddingGemma2Model.get_image_features in the Hugging Face
 * reference: the vision tower emits padded patch states; the multimodal
 * embedder projects them; valid patch positions determine each image's
 * unpadded soft-token count. The pooling kernel determines how many patch
 * positions correspond to one output soft token.</p>
 *
 * <p>Position coordinates use {@code -1} for padding. A position is valid only
 * when both coordinates are non-negative.</p>
 */
public final class EmbeddingGemma2VisionTokenSplitter {
    private EmbeddingGemma2VisionTokenSplitter() {
    }

    /**
     * @param projectedStates flattened [totalSoftTokens][textHiddenSize] states
     * @param imagePositionIds [images][maxPatches][2], with (-1,-1) padding
     * @param paddedPatchCount number of patch positions in pixel_values' patch axis
     * @param poolingKernelSize vision tower spatial pooling kernel
     */
    public static List<float[][]> split(
            float[][] projectedStates,
            int[][][] imagePositionIds,
            int paddedPatchCount,
            int poolingKernelSize) {
        Objects.requireNonNull(projectedStates, "projectedStates must not be null");
        Objects.requireNonNull(imagePositionIds, "imagePositionIds must not be null");
        if (projectedStates.length == 0) {
            throw new IllegalArgumentException("projectedStates must contain soft tokens");
        }
        if (imagePositionIds.length == 0) {
            throw new IllegalArgumentException("at least one image is required");
        }
        if (paddedPatchCount <= 0 || poolingKernelSize <= 0) {
            throw new IllegalArgumentException("patch count and pooling kernel must be positive");
        }
        long kernelArea = (long) poolingKernelSize * poolingKernelSize;
        if (paddedPatchCount % kernelArea != 0) {
            throw new IllegalArgumentException("padded patch count must be divisible by pooling kernel area");
        }
        int outputLength = (int) (paddedPatchCount / kernelArea);
        int maxPatches = validatePositions(imagePositionIds);
        if (maxPatches % outputLength != 0) {
            throw new IllegalArgumentException("position width must be divisible by vision output length");
        }
        int ratio = maxPatches / outputLength;
        int spatialGroup = (int) Math.sqrt(ratio);
        if (spatialGroup * spatialGroup != ratio) {
            throw new IllegalArgumentException("position/output ratio must be a perfect square");
        }
        int patchesPerSoftToken = spatialGroup * spatialGroup;

        int[] sizes = new int[imagePositionIds.length];
        int total = 0;
        for (int image = 0; image < imagePositionIds.length; image++) {
            int valid = 0;
            for (int[] position : imagePositionIds[image]) {
                if (position[0] >= 0 && position[1] >= 0) {
                    valid++;
                } else if (position[0] != -1 || position[1] != -1) {
                    throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
                }
            }
            if (valid % patchesPerSoftToken != 0) {
                throw new IllegalArgumentException("valid patch count for image " + image
                        + " is not divisible by patches per soft token");
            }
            sizes[image] = valid / patchesPerSoftToken;
            if (sizes[image] == 0) {
                throw new IllegalArgumentException("image " + image + " has no valid soft tokens");
            }
            total = Math.addExact(total, sizes[image]);
        }
        if (total != projectedStates.length) {
            throw new IllegalArgumentException("projected soft-token count " + projectedStates.length
                    + " does not match image position metadata total " + total);
        }

        List<float[][]> result = new ArrayList<>(sizes.length);
        int offset = 0;
        int hiddenSize = validateProjectedStates(projectedStates);
        for (int size : sizes) {
            float[][] imageStates = new float[size][hiddenSize];
            for (int token = 0; token < size; token++) {
                System.arraycopy(projectedStates[offset++], 0, imageStates[token], 0, hiddenSize);
            }
            result.add(imageStates);
        }
        return List.copyOf(result);
    }

    private static int validatePositions(int[][][] positions) {
        if (positions[0] == null || positions[0].length == 0) {
            throw new IllegalArgumentException("image position rows must be non-empty");
        }
        int width = positions[0].length;
        for (int image = 0; image < positions.length; image++) {
            if (positions[image] == null || positions[image].length != width) {
                throw new IllegalArgumentException("image position tensor must be rectangular");
            }
            for (int patch = 0; patch < width; patch++) {
                if (positions[image][patch] == null || positions[image][patch].length != 2) {
                    throw new IllegalArgumentException("each image position must contain exactly (x,y)");
                }
            }
        }
        return width;
    }

    private static int validateProjectedStates(float[][] states) {
        if (states[0] == null || states[0].length == 0) {
            throw new IllegalArgumentException("projected states must have a non-empty hidden dimension");
        }
        int width = states[0].length;
        for (float[] state : states) {
            if (state == null || state.length != width) {
                throw new IllegalArgumentException("projected states must be rectangular");
            }
            for (float value : state) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("projected states must contain finite values");
                }
            }
        }
        return width;
    }
}
