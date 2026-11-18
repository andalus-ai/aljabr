package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Gemma 4 vision patch projection and learned two-axis position embedding.
 *
 * <p>Consumes already extracted RGB patch vectors. Image resizing, RGB
 * conversion, patch extraction, and patch ordering belong to the processor;
 * this class implements the checkpoint-backed model operation:
 * {@code 2 * (pixel - 0.5)} followed by a bias-free patch projection and the
 * sum of learned x/y position embeddings.</p>
 *
 * <p>Input patch vectors use row-major flattened RGB values with shape
 * {@code [patches][3 * patchSize * patchSize]}. Position IDs use
 * {@code [patches][2]} in (x,y) order. Padding is exactly {@code (-1,-1)} and
 * contributes no positional embedding.</p>
 */
public final class EmbeddingGemma2VisionPatchEmbedder {
    private final int patchSize;
    private final int hiddenSize;
    private final int positionEmbeddingSize;
    private final float[][] inputProjection;
    private final float[][][] positionEmbeddingTable;

    public EmbeddingGemma2VisionPatchEmbedder(
            int patchSize,
            int hiddenSize,
            int positionEmbeddingSize,
            Weights weights) {
        if (patchSize <= 0 || hiddenSize <= 0 || positionEmbeddingSize <= 0) {
            throw new IllegalArgumentException("patch size and embedding dimensions must be positive");
        }
        Objects.requireNonNull(weights, "weights must not be null");
        this.patchSize = patchSize;
        this.hiddenSize = hiddenSize;
        this.positionEmbeddingSize = positionEmbeddingSize;
        int patchWidth = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        this.inputProjection = copyMatrix(weights.inputProjection(), hiddenSize, patchWidth,
                "inputProjection");
        this.positionEmbeddingTable = copyPositionTable(
                weights.positionEmbeddingTable(), positionEmbeddingSize, hiddenSize);
    }

    /**
     * Embeds one image's patches.
     *
     * @param patchPixels flattened RGB patch values in processor order
     * @param positionIds patch coordinates as (x,y), or (-1,-1) for padding
     * @return patch embeddings with shape [patches][hiddenSize]
     */
    public float[][] embed(float[][] patchPixels, int[][] positionIds) {
        Objects.requireNonNull(patchPixels, "patchPixels must not be null");
        Objects.requireNonNull(positionIds, "positionIds must not be null");
        if (patchPixels.length == 0 || patchPixels.length != positionIds.length) {
            throw new IllegalArgumentException("patch pixels and position IDs must have the same non-zero length");
        }
        int patchWidth = inputProjection[0].length;
        float[][] output = new float[patchPixels.length][hiddenSize];
        for (int patch = 0; patch < patchPixels.length; patch++) {
            float[] pixels = patchPixels[patch];
            int[] position = positionIds[patch];
            if (pixels == null || pixels.length != patchWidth) {
                throw new IllegalArgumentException("patch " + patch + " must contain " + patchWidth + " RGB values");
            }
            if (position == null || position.length != 2) {
                throw new IllegalArgumentException("patch " + patch + " position must contain exactly (x,y)");
            }
            boolean padding = position[0] == -1 && position[1] == -1;
            if (!padding && (position[0] < 0 || position[1] < 0)) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
            if (!padding && (position[0] >= positionEmbeddingSize || position[1] >= positionEmbeddingSize)) {
                throw new IllegalArgumentException("patch " + patch + " position exceeds learned embedding table");
            }

            for (int hidden = 0; hidden < hiddenSize; hidden++) {
                float sum = 0f;
                for (int channel = 0; channel < patchWidth; channel++) {
                    float pixel = pixels[channel];
                    if (!Float.isFinite(pixel)) {
                        throw new IllegalArgumentException("patch pixels must contain finite values");
                    }
                    // The reference scales [0,1] pixels to [-1,1] before projection.
                    float scaled = 2f * (pixel - 0.5f);
                    sum += scaled * inputProjection[hidden][channel];
                }
                if (!padding) {
                    sum += positionEmbeddingTable[0][position[0]][hidden];
                    sum += positionEmbeddingTable[1][position[1]][hidden];
                }
                output[patch][hidden] = sum;
            }
        }
        return output;
    }

    public int patchSize() {
        return patchSize;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int positionEmbeddingSize() {
        return positionEmbeddingSize;
    }

    private static float[][] copyMatrix(float[][] source, int rows, int columns, String name) {
        if (source == null || source.length != rows) {
            throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
        }
        float[][] copy = new float[rows][columns];
        for (int row = 0; row < rows; row++) {
            if (source[row] == null || source[row].length != columns) {
                throw new IllegalArgumentException(name + " must have shape [" + rows + "][" + columns + "]");
            }
            for (int column = 0; column < columns; column++) {
                float value = source[row][column];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException(name + " must contain finite values");
                }
                copy[row][column] = value;
            }
        }
        return copy;
    }

    private static float[][][] copyPositionTable(float[][][] source, int positions, int hiddenSize) {
        if (source == null || source.length != 2) {
            throw new IllegalArgumentException("positionEmbeddingTable must have shape [2]["
                    + positions + "][" + hiddenSize + "]");
        }
        float[][][] copy = new float[2][positions][hiddenSize];
        for (int axis = 0; axis < 2; axis++) {
            if (source[axis] == null || source[axis].length != positions) {
                throw new IllegalArgumentException("positionEmbeddingTable must have shape [2]["
                        + positions + "][" + hiddenSize + "]");
            }
            for (int position = 0; position < positions; position++) {
                if (source[axis][position] == null || source[axis][position].length != hiddenSize) {
                    throw new IllegalArgumentException("positionEmbeddingTable must have shape [2]["
                            + positions + "][" + hiddenSize + "]");
                }
                for (int hidden = 0; hidden < hiddenSize; hidden++) {
                    float value = source[axis][position][hidden];
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("positionEmbeddingTable must contain finite values");
                    }
                    copy[axis][position][hidden] = value;
                }
            }
        }
        return copy;
    }

    /** Checkpoint tensors in [out,in] and [axis,position,hidden] layouts. */
    public record Weights(float[][] inputProjection, float[][][] positionEmbeddingTable) {
        public Weights {
            Objects.requireNonNull(inputProjection, "inputProjection must not be null");
            Objects.requireNonNull(positionEmbeddingTable, "positionEmbeddingTable must not be null");
        }
    }
}
