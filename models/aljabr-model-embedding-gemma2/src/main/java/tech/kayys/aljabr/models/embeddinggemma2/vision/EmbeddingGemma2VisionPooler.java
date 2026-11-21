package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.Objects;

/**
 * Gemma4's position-aware spatial average pooler and sqrt(hiddenSize) scaling.
 *
 * <p>Pooling groups patches into k-by-k spatial cells using their (x,y)
 * coordinates. Every cell is divided by k^2, including cells containing
 * padding; padding states are zeroed before aggregation, matching the
 * checkpoint reference's fixed-area average. Results are promoted to
 * float32 for the final sqrt(hiddenSize) scale.</p>
 */
public final class EmbeddingGemma2VisionPooler {
    private final int hiddenSize;
    private final float rootHiddenSize;

    public EmbeddingGemma2VisionPooler(int hiddenSize) {
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive");
        }
        this.hiddenSize = hiddenSize;
        this.rootHiddenSize = (float) Math.sqrt(hiddenSize);
    }

    /**
     * Pools one image's final patch states.
     *
     * @param hiddenStates final encoder states [patches][hiddenSize]
     * @param pixelPositionIds patch coordinates (x,y); (-1,-1) denotes padding
     * @param paddingPositions true for padded patch entries
     * @param outputLength desired number of spatial soft tokens
     * @return pooled states and true=padding output mask
     */
    public Result pool(
            float[][] hiddenStates,
            int[][] pixelPositionIds,
            boolean[] paddingPositions,
            int outputLength) {
        validate(hiddenStates, pixelPositionIds, paddingPositions, outputLength);

        int inputLength = hiddenStates.length;
        float[][] pooled;
        boolean[] outputPadding;

        if (inputLength == outputLength) {
            pooled = copyAndZeroPadding(hiddenStates, paddingPositions);
            outputPadding = new boolean[inputLength];
            for (int i = 0; i < inputLength; i++) {
                outputPadding[i] = paddingPositions[i];
            }
        } else {
            int kernel = (int) Math.sqrt((double) inputLength / outputLength);
            int kernelArea = kernel * kernel;
            if (kernel <= 0 || kernelArea * outputLength != inputLength) {
                throw new IllegalArgumentException(
                        "patch count must equal outputLength * poolingKernelSize^2");
            }

            int maxX = 0;
            for (int i = 0; i < inputLength; i++) {
                if (!paddingPositions[i]) {
                    maxX = Math.max(maxX, pixelPositionIds[i][0] + 1);
                }
            }
            int maxY = 0;
            int validPatchCount = 0;
            for (int patch = 0; patch < inputLength; patch++) {
                if (!paddingPositions[patch]) {
                    maxY = Math.max(maxY, pixelPositionIds[patch][1] + 1);
                    validPatchCount++;
                }
            }
            if (maxX <= 0 || maxY <= 0 || maxX % kernel != 0 || maxY % kernel != 0) {
                throw new IllegalArgumentException(
                        "valid patch grid dimensions must be divisible by the spatial pooling kernel");
            }
            int cellsPerRow = maxX / kernel;
            if (cellsPerRow <= 0) {
                throw new IllegalArgumentException("valid patch positions do not define a pooling grid");
            }
            for (int patch = 0; patch < validPatchCount; patch++) {
                int expectedX = patch % maxX;
                int expectedY = patch / maxX;
                if (pixelPositionIds[patch][0] != expectedX || pixelPositionIds[patch][1] != expectedY) {
                    throw new IllegalArgumentException(
                            "valid patch positions must form a row-major rectangular grid");
                }
            }

            pooled = new float[outputLength][hiddenSize];
            boolean[] hasValidPatch = new boolean[outputLength];
            for (int patch = 0; patch < inputLength; patch++) {
                int x = Math.max(0, pixelPositionIds[patch][0]);
                int y = Math.max(0, pixelPositionIds[patch][1]);
                int cell = (x / kernel) + cellsPerRow * (y / kernel);
                if (cell < 0 || cell >= outputLength) {
                    throw new IllegalArgumentException(
                            "patch position maps outside the configured pooled sequence at patch " + patch);
                }
                if (!paddingPositions[patch]) {
                    hasValidPatch[cell] = true;
                    for (int d = 0; d < hiddenSize; d++) {
                        pooled[cell][d] += hiddenStates[patch][d] / kernelArea;
                    }
                }
            }
            outputPadding = new boolean[outputLength];
            for (int i = 0; i < outputLength; i++) {
                outputPadding[i] = !hasValidPatch[i];
            }
        }

        for (int token = 0; token < outputLength; token++) {
            for (int d = 0; d < hiddenSize; d++) {
                pooled[token][d] *= rootHiddenSize;
            }
        }
        return new Result(pooled, outputPadding);
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    private void validate(
            float[][] states, int[][] positions, boolean[] padding, int outputLength) {
        Objects.requireNonNull(states, "hiddenStates must not be null");
        Objects.requireNonNull(positions, "pixelPositionIds must not be null");
        Objects.requireNonNull(padding, "paddingPositions must not be null");
        if (states.length == 0 || positions.length != states.length || padding.length != states.length) {
            throw new IllegalArgumentException("states, positions, and padding mask must have equal non-zero length");
        }
        if (outputLength <= 0 || outputLength > states.length) {
            throw new IllegalArgumentException("outputLength must be in [1, patchCount]");
        }
        boolean valid = false;
        for (int i = 0; i < states.length; i++) {
            if (states[i] == null || states[i].length != hiddenSize) {
                throw new IllegalArgumentException("hiddenStates must be rectangular with hiddenSize columns");
            }
            for (float value : states[i]) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("hiddenStates must contain only finite values");
                }
            }
            if (positions[i] == null || positions[i].length != 2) {
                throw new IllegalArgumentException("pixel position " + i + " must contain (x,y)");
            }
            boolean sentinel = positions[i][0] == -1 && positions[i][1] == -1;
            if ((positions[i][0] < 0 || positions[i][1] < 0) && !sentinel) {
                throw new IllegalArgumentException("padding position must be exactly (-1,-1)");
            }
            if (sentinel != padding[i]) {
                throw new IllegalArgumentException("padding mask must agree with (-1,-1) position sentinel");
            }
            valid |= !padding[i];
        }
        if (!valid) {
            throw new IllegalArgumentException("at least one valid patch is required");
        }
    }

    private static float[][] copyAndZeroPadding(float[][] states, boolean[] padding) {
        float[][] copy = new float[states.length][];
        for (int i = 0; i < states.length; i++) {
            copy[i] = states[i].clone();
            if (padding[i]) {
                java.util.Arrays.fill(copy[i], 0f);
            }
        }
        return copy;
    }

    /** Immutable result wrapper with defensive copies. */
    public static final class Result {
        private final float[][] hiddenStates;
        private final boolean[] paddingMask;

        private Result(float[][] hiddenStates, boolean[] paddingMask) {
            this.hiddenStates = deepCopy(hiddenStates);
            this.paddingMask = paddingMask.clone();
        }

        public float[][] hiddenStates() {
            return deepCopy(hiddenStates);
        }

        public boolean[] paddingMask() {
            return paddingMask.clone();
        }

        private static float[][] deepCopy(float[][] values) {
            float[][] copy = new float[values.length][];
            for (int i = 0; i < values.length; i++) copy[i] = values[i].clone();
            return copy;
        }
    }
}
