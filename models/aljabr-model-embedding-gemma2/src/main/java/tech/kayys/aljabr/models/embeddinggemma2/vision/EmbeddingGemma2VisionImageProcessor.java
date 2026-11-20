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
     * Separable bicubic resize with antialiasing, matching torchvision's
     * tensor backend and ATen's Pillow-compatible Keys cubic filter (a=-0.5).
     * Filter support expands when downsampling; taps outside the input are
     * excluded and the remaining weights are normalized.
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
                    // Tensor bicubic interpolation preserves cubic overshoot;
                    // do not clamp before the processor's 1/255 rescale.
                    output[(y * targetWidth + x) * 3 + channel] = (float) value;
                }
            }
        }
        return RgbImage.fromResizedFloatSamples(targetWidth, targetHeight, output);
    }

    private static Contributions contributions(int inputSize, int outputSize, int outputIndex) {
        double scale = (double) inputSize / outputSize;
        double support = scale >= 1d ? 2d * scale : 2d;
        double inverseScale = scale >= 1d ? 1d / scale : 1d;
        double center = scale * (outputIndex + 0.5d);

        // Mirrors ATen HelperInterpBase::_compute_indices_min_size_weights_aa.
        // Java's cast from finite double to int truncates toward zero, as C++ does.
        int first = Math.max((int) (center - support + 0.5d), 0);
        int end = Math.min((int) (center + support + 0.5d), inputSize);
        int length = Math.max(0, end - first);
        int[] indices = new int[length];
        double[] weights = new double[length];
        double sum = 0d;
        for (int i = 0; i < length; i++) {
            indices[i] = first + i;
            weights[i] = cubic((i + first - center + 0.5d) * inverseScale);
            sum += weights[i];
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

        /** Creates RGB by dropping alpha from interleaved RGBA8 samples, as PIL RGB conversion does. */
        public static RgbImage fromRgba8(int width, int height, byte[] rgba) {
            Objects.requireNonNull(rgba, "rgba must not be null");
            int pixels = Math.multiplyExact(width, height);
            if (rgba.length != Math.multiplyExact(pixels, 4)) {
                throw new IllegalArgumentException("RGBA sample count must equal width * height * 4");
            }
            byte[] rgb = new byte[Math.multiplyExact(pixels, 3)];
            for (int pixel = 0; pixel < pixels; pixel++) {
                rgb[pixel * 3] = rgba[pixel * 4];
                rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
                rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
            }
            return fromRgb8(width, height, rgb);
        }

        /** Creates RGB by replicating each unsigned grayscale sample into R, G, and B. */
        public static RgbImage fromGrayscale8(int width, int height, byte[] grayscale) {
            Objects.requireNonNull(grayscale, "grayscale must not be null");
            int pixels = Math.multiplyExact(width, height);
            if (grayscale.length != pixels) {
                throw new IllegalArgumentException("grayscale sample count must equal width * height");
            }
            byte[] rgb = new byte[Math.multiplyExact(pixels, 3)];
            for (int pixel = 0; pixel < pixels; pixel++) {
                rgb[pixel * 3] = grayscale[pixel];
                rgb[pixel * 3 + 1] = grayscale[pixel];
                rgb[pixel * 3 + 2] = grayscale[pixel];
            }
            return fromRgb8(width, height, rgb);
        }

        /** Creates RGB by reordering interleaved BGR8 samples. */
        public static RgbImage fromBgr8(int width, int height, byte[] bgr) {
            Objects.requireNonNull(bgr, "bgr must not be null");
            int pixels = Math.multiplyExact(width, height);
            if (bgr.length != Math.multiplyExact(pixels, 3)) {
                throw new IllegalArgumentException("BGR sample count must equal width * height * 3");
            }
            byte[] rgb = new byte[bgr.length];
            for (int pixel = 0; pixel < pixels; pixel++) {
                rgb[pixel * 3] = bgr[pixel * 3 + 2];
                rgb[pixel * 3 + 1] = bgr[pixel * 3 + 1];
                rgb[pixel * 3 + 2] = bgr[pixel * 3];
            }
            return fromRgb8(width, height, rgb);
        }

        /** Creates an image from interleaved RGB float samples in [0,255]. */
        public static RgbImage fromFloatSamples(int width, int height, float[] rgb) {
            return new RgbImage(width, height, rgb);
        }

        public int width() { return width; }
        public int height() { return height; }

        /** Returns a defensive RGB triplet in [0,255] at the requested pixel. */
        public float[] rgbAt(int x, int y) {
            if (x < 0 || x >= width || y < 0 || y >= height) {
                throw new IndexOutOfBoundsException("pixel coordinate is outside the image");
            }
            int offset = (y * width + x) * 3;
            return new float[]{samples[offset], samples[offset + 1], samples[offset + 2]};
        }

        private float channel(int index) { return samples[index]; }
    }
}
