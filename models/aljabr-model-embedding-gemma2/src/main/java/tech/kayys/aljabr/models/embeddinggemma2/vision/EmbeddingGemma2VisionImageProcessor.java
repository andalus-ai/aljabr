package tech.kayys.aljabr.models.embeddinggemma2.vision;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Java RGB image preprocessing for the Gemma4 vision path used by
 * EmbeddingGemma 2.
 *
 * <p>The resize budget, dimension rounding, bicubic antialias filter,
 * patch layout, position IDs, and padding follow Hugging Face's
 * {@code Gemma4ImageProcessor}. This class has no AWT, image-codec, or
 * third-party image dependency: callers decode images and pass packed RGB
 * samples through {@link RgbImage}.</p>
 *
 * <p>Input samples are unsigned 8-bit RGB values (or equivalent floats in
 * [0,255]). Resized samples are clamped to [0,255], rescaled by 1/255, and
 * patchified in row-major order. Within each patch, values are ordered
 * y, x, channel (RGB), matching the processor's NCHW-to-patch permutation.</p>
 */
public final class EmbeddingGemma2VisionImageProcessor {
    private static final int[] SUPPORTED_SOFT_TOKENS = {70, 140, 280, 560, 1120};
    private final int patchSize;
    private final int maxSoftTokens;
    private final int poolingKernelSize;
    private final int maxPatches;

    public EmbeddingGemma2VisionImageProcessor(
            int patchSize, int maxSoftTokens, int poolingKernelSize) {
        if (patchSize <= 0 || poolingKernelSize <= 0) {
            throw new IllegalArgumentException("patchSize and poolingKernelSize must be positive");
        }
        boolean supported = false;
        for (int value : SUPPORTED_SOFT_TOKENS) supported |= value == maxSoftTokens;
        if (!supported) {
            throw new IllegalArgumentException("maxSoftTokens must be one of "
                    + List.of(70, 140, 280, 560, 1120) + ", got " + maxSoftTokens);
        }
        this.patchSize = patchSize;
        this.maxSoftTokens = maxSoftTokens;
        this.poolingKernelSize = poolingKernelSize;
        this.maxPatches = Math.multiplyExact(maxSoftTokens,
                Math.multiplyExact(poolingKernelSize, poolingKernelSize));
    }

    /** Processes one image and returns the same padded tensor contract as HF. */
    public EmbeddingGemma2VisionProcessorOutput process(RgbImage image) {
        return process(List.of(Objects.requireNonNull(image, "image must not be null")));
    }

    /** Processes a batch with independent aspect-ratio-preserving resize per image. */
    public EmbeddingGemma2VisionProcessorOutput process(List<RgbImage> images) {
        Objects.requireNonNull(images, "images must not be null");
        if (images.isEmpty()) throw new IllegalArgumentException("at least one image is required");

        float[][][] batchPixels = new float[images.size()][maxPatches][];
        int[][][] batchPositions = new int[images.size()][maxPatches][2];
        int[] softTokens = new int[images.size()];
        for (int index = 0; index < images.size(); index++) {
            RgbImage image = Objects.requireNonNull(images.get(index), "image " + index + " must not be null");
            int[] target = aspectRatioPreservingSize(image.height(), image.width());
            RgbImage resized = target[0] == image.height() && target[1] == image.width()
                    ? image : resizeBicubicAntialias(image, target[0], target[1]);
            patchify(resized, batchPixels[index], batchPositions[index], index, softTokens);
        }
        return new EmbeddingGemma2VisionProcessorOutput(batchPixels, batchPositions, softTokens);
    }

    public int patchSize() { return patchSize; }
    public int maxSoftTokens() { return maxSoftTokens; }
    public int poolingKernelSize() { return poolingKernelSize; }
    public int maxPatches() { return maxPatches; }

    /**
     * Gemma4's floor-to-grid aspect-ratio resize calculation.
     * Returned dimensions are multiples of poolingKernelSize * patchSize.
     */
    public int[] aspectRatioPreservingSize(int height, int width) {
        if (height <= 0 || width <= 0) throw new IllegalArgumentException("image dimensions must be positive");
        double totalPixels = (double) height * width;
        double targetPixels = (double) maxPatches * patchSize * patchSize;
        double factor = Math.sqrt(targetPixels / totalPixels);
        int sideMultiple = Math.multiplyExact(poolingKernelSize, patchSize);
        int targetHeight = floorToMultiple(factor * height, sideMultiple);
        int targetWidth = floorToMultiple(factor * width, sideMultiple);

        if (targetHeight == 0 && targetWidth == 0) {
            throw new IllegalArgumentException("image is too small to resize to a non-zero processor grid");
        }
        int maxSideLength = Math.multiplyExact(maxSoftTokens, sideMultiple);
        if (targetHeight == 0) {
            targetHeight = sideMultiple;
            targetWidth = Math.min(floorToMultiple((double) width / height * sideMultiple, sideMultiple),
                    maxSideLength);
        } else if (targetWidth == 0) {
            targetWidth = sideMultiple;
            targetHeight = Math.min(floorToMultiple((double) height / width * sideMultiple, sideMultiple),
                    maxSideLength);
        }
        if ((long) targetHeight * targetWidth > targetPixels + 1e-7) {
            throw new IllegalStateException("computed resize exceeds the configured patch budget");
        }
        return new int[]{targetHeight, targetWidth};
    }

    private static int floorToMultiple(double value, int multiple) {
        if (!Double.isFinite(value) || value < 0) throw new IllegalArgumentException("invalid resize dimension");
        double units = Math.floor(value / multiple);
        if (units > Integer.MAX_VALUE / (double) multiple) {
            throw new IllegalArgumentException("resized image dimension overflows an integer");
        }
        return (int) units * multiple;
    }

    private void patchify(
            RgbImage image,
            float[][] paddedPatches,
            int[][] paddedPositions,
            int imageIndex,
            int[] softTokens) {
        int patchRows = image.height() / patchSize;
        int patchColumns = image.width() / patchSize;
        int patchCount = Math.multiplyExact(patchRows, patchColumns);
        int patchWidth = Math.multiplyExact(3, Math.multiplyExact(patchSize, patchSize));
        if (patchCount > maxPatches || patchCount % (poolingKernelSize * poolingKernelSize) != 0) {
            throw new IllegalStateException("resized image does not satisfy the configured patch budget/grid");
        }
        softTokens[imageIndex] = patchCount / (poolingKernelSize * poolingKernelSize);

        for (int patchY = 0; patchY < patchRows; patchY++) {
            for (int patchX = 0; patchX < patchColumns; patchX++) {
                int patchIndex = patchY * patchColumns + patchX;
                float[] patch = new float[patchWidth];
                int cursor = 0;
                for (int localY = 0; localY < patchSize; localY++) {
                    for (int localX = 0; localX < patchSize; localX++) {
                        int pixelOffset = ((patchY * patchSize + localY) * image.width()
                                + patchX * patchSize + localX) * 3;
                        patch[cursor++] = image.channel(pixelOffset) / 255f;
                        patch[cursor++] = image.channel(pixelOffset + 1) / 255f;
                        patch[cursor++] = image.channel(pixelOffset + 2) / 255f;
                    }
                }
                paddedPatches[patchIndex] = patch;
                paddedPositions[patchIndex][0] = patchX;
                paddedPositions[patchIndex][1] = patchY;
            }
        }
        for (int patch = patchCount; patch < maxPatches; patch++) {
            paddedPatches[patch] = new float[patchWidth];
            paddedPositions[patch][0] = -1;
            paddedPositions[patch][1] = -1;
        }
    }

    /**
     * Separable bicubic resize with antialiasing. The cubic coefficient is
     * -0.5 (Pillow/torchvision bicubic); the support expands when downsampling.
     */
    private static RgbImage resizeBicubicAntialias(RgbImage source, int targetHeight, int targetWidth) {
        float[] horizontal = new float[Math.multiplyExact(
                Math.multiplyExact(source.height(), targetWidth), 3)];
        for (int y = 0; y < source.height(); y++) {
            for (int x = 0; x < targetWidth; x++) {
                Contributions contribution = contributions(source.width(), targetWidth, x);
                for (int channel = 0; channel < 3; channel++) {
                    double value = 0d;
                    for (int i = 0; i < contribution.indices.length; i++) {
                        value += source.channel((y * source.width() + contribution.indices[i]) * 3 + channel)
                                * contribution.weights[i];
                    }
                    horizontal[(y * targetWidth + x) * 3 + channel] = (float) value;
                }
            }
        }

        float[] output = new float[Math.multiplyExact(Math.multiplyExact(targetHeight, targetWidth), 3)];
        for (int y = 0; y < targetHeight; y++) {
            Contributions contribution = contributions(source.height(), targetHeight, y);
            for (int x = 0; x < targetWidth; x++) {
                for (int channel = 0; channel < 3; channel++) {
                    double value = 0d;
                    for (int i = 0; i < contribution.indices.length; i++) {
                        value += horizontal[(contribution.indices[i] * targetWidth + x) * 3 + channel]
                                * contribution.weights[i];
                    }
                    output[(y * targetWidth + x) * 3 + channel] =
                            (float) Math.max(0d, Math.min(255d, value));
                }
            }
        }
        return RgbImage.fromFloatSamples(targetWidth, targetHeight, output);
    }

    private static Contributions contributions(int inputSize, int outputSize, int outputIndex) {
        double scale = (double) inputSize / outputSize;
        double filterScale = Math.max(1d, scale);
        double center = (outputIndex + 0.5d) * scale;
        int first = Math.max(0, (int) Math.floor(center - 2d * filterScale - 0.5d));
        int last = Math.min(inputSize - 1, (int) Math.ceil(center + 2d * filterScale - 0.5d));
        int length = last - first + 1;
        int[] indices = new int[length];
        double[] weights = new double[length];
        double sum = 0d;
        for (int i = 0; i < length; i++) {
            int sample = first + i;
            double distance = (sample + 0.5d - center) / filterScale;
            double weight = cubic(distance);
            indices[i] = sample;
            weights[i] = weight;
            sum += weight;
        }
        if (sum == 0d) throw new IllegalStateException("bicubic resize produced an empty filter");
        for (int i = 0; i < length; i++) weights[i] /= sum;
        return new Contributions(indices, weights);
    }

    private static double cubic(double x) {
        final double a = -0.5d;
        x = Math.abs(x);
        if (x < 1d) return ((a + 2d) * x - (a + 3d)) * x * x + 1d;
        if (x < 2d) return (((a * x - 5d * a) * x + 8d * a) * x) - 4d * a;
        return 0d;
    }

    private record Contributions(int[] indices, double[] weights) {}

    /** Immutable packed RGB image; samples are stored as interleaved RGB floats in [0,255]. */
    public static final class RgbImage {
        private final int width;
        private final int height;
        private final float[] samples;

        private RgbImage(int width, int height, float[] samples) {
            if (width <= 0 || height <= 0) throw new IllegalArgumentException("image dimensions must be positive");
            if (samples == null || samples.length != Math.multiplyExact(Math.multiplyExact(width, height), 3)) {
                throw new IllegalArgumentException("RGB sample count must equal width * height * 3");
            }
            this.width = width;
            this.height = height;
            this.samples = samples.clone();
            for (float sample : this.samples) {
                if (!Float.isFinite(sample) || sample < 0f || sample > 255f) {
                    throw new IllegalArgumentException("RGB samples must be finite values in [0,255]");
                }
            }
        }

        /** Creates an image from interleaved unsigned-byte RGB samples. */
        public static RgbImage fromRgb8(int width, int height, byte[] rgb) {
            Objects.requireNonNull(rgb, "rgb must not be null");
            float[] values = new float[rgb.length];
            for (int i = 0; i < rgb.length; i++) values[i] = rgb[i] & 0xff;
            return new RgbImage(width, height, values);
        }

        /** Creates an image from interleaved RGB float samples in [0,255]. */
        public static RgbImage fromFloatSamples(int width, int height, float[] rgb) {
            return new RgbImage(width, height, rgb);
        }

        public int width() { return width; }
        public int height() { return height; }
        private float channel(int index) { return samples[index]; }
    }
}
