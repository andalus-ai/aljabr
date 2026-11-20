package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionImageProcessorTest {
    private static final int PATCH_SIZE = 2;
    private static final int POOLING_KERNEL = 2;
    private static final int MAX_SOFT_TOKENS = 70;

    @Test
    void usesGemma4FloorToPoolingPatchGridForAspectPreservingResize() {
        var processor = processor();

        int[] size = processor.aspectRatioPreservingSize(100, 200);

        assertEquals(20, size[0]);
        assertEquals(44, size[1]);
        assertEquals(0, size[0] % (PATCH_SIZE * POOLING_KERNEL));
        assertEquals(0, size[1] % (PATCH_SIZE * POOLING_KERNEL));
        assertEquals(220, (size[0] / PATCH_SIZE) * (size[1] / PATCH_SIZE));
    }

    @Test
    void patchifiesInRowMajorPatchOrderAndYxRgbPixelOrder() {
        byte[] rgb = new byte[4 * 280 * 3];
        for (int i = 0; i < rgb.length; i++) rgb[i] = (byte) (i & 0xff);
        var image = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(280, 4, rgb);

        var output = processor().process(image);
        float[][] patches = output.pixelValues()[0];
        int[][] positions = output.imagePositionIds()[0];

        assertEquals(70, output.numSoftTokensPerImage()[0]);
        assertEquals(280, patches.length);
        assertEquals(12, patches[0].length);
        assertEquals(0, positions[0][0]);
        assertEquals(0, positions[0][1]);
        assertEquals(1, positions[1][0]);
        assertEquals(0, positions[1][1]);
        assertEquals(0, positions[140][0]);
        assertEquals(1, positions[140][1]);

        int[] expectedSourceOffsets = {0, 3, 840, 843};
        for (int pixel = 0; pixel < expectedSourceOffsets.length; pixel++) {
            int offset = expectedSourceOffsets[pixel];
            for (int channel = 0; channel < 3; channel++) {
                assertEquals((rgb[offset + channel] & 0xff) / 255f,
                        patches[0][pixel * 3 + channel], 1e-7f);
            }
        }
        assertEquals(139, positions[279][0]);
        assertEquals(1, positions[279][1]);
    }

    @Test
    void resizesAndPadsDifferentAspectRatioImagesToSharedPatchAxis() {
        byte[] firstPixels = new byte[100 * 200 * 3];
        Arrays.fill(firstPixels, (byte) 127);
        byte[] secondPixels = new byte[200 * 100 * 3];
        Arrays.fill(secondPixels, (byte) 64);
        var first = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(200, 100, firstPixels);
        var second = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(100, 200, secondPixels);

        var output = processor().process(java.util.List.of(first, second));
        float[][][] pixels = output.pixelValues();
        int[][][] positions = output.imagePositionIds();

        assertEquals(2, output.imageCount());
        assertEquals(280, output.paddedPatchCount());
        assertEquals(55, output.numSoftTokensPerImage()[0]);
        assertEquals(55, output.numSoftTokensPerImage()[1]);
        assertEquals(127f / 255f, pixels[0][0][0], 1e-5f);
        assertEquals(64f / 255f, pixels[1][0][0], 1e-5f);
        assertEquals(21, positions[0][219][0]);
        assertEquals(9, positions[0][219][1]);
        assertEquals(-1, positions[0][220][0]);
        assertEquals(9, positions[1][219][0]);
        assertEquals(21, positions[1][219][1]);
        assertEquals(-1, positions[1][220][1]);
    }

    @Test
    void matchesTorchvisionBicubicAntialiasReferenceFixtureIncludingOvershoot() {
        byte[] rgb = new byte[7 * 9 * 3];
        for (int i = 0; i < rgb.length; i++) rgb[i] = (byte) (i & 0xff);
        var image = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(9, 7, rgb);

        var output = processor().process(image);
        float[] actual = output.pixelValues()[0][0];

        // Reference generated with torchvision.transforms.v2.functional.resize:
        // [1,3,7,9] -> [1,3,28,36], BICUBIC, antialias=True, then /255.
        // Patch layout is [y, x, channel] for the top-left 2x2 patch.
        float[] expected = {
                -0.013169447f, -0.009247878f, -0.005326311f,
                -0.012467078f, -0.008545508f, -0.004623939f,
                -0.006848112f, -0.002926543f,  0.000995024f,
                -0.006145743f, -0.002224173f,  0.001697396f
        };
        assertEquals(63, output.numSoftTokensPerImage()[0]);
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 2e-6f, "pixel " + i);
        }
        // Bicubic filtering legitimately overshoots below zero; the adapter must preserve it.
        org.junit.jupiter.api.Assertions.assertTrue(actual[0] < 0f);
    }

    @Test
    void convertsGrayscaleAndRgbaInputsToRgbWithoutAlphaCompositing() {
        var gray = EmbeddingGemma2VisionImageProcessor.RgbImage.fromGrayscale8(
                1, 1, new byte[]{(byte) 128});
        assertEquals(128f, gray.rgbAt(0, 0)[0], 0f);
        assertEquals(128f, gray.rgbAt(0, 0)[1], 0f);
        assertEquals(128f, gray.rgbAt(0, 0)[2], 0f);

        var rgba = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgba8(
                1, 1, new byte[]{10, 20, 30, 0});
        assertEquals(10f, rgba.rgbAt(0, 0)[0], 0f);
        assertEquals(20f, rgba.rgbAt(0, 0)[1], 0f);
        assertEquals(30f, rgba.rgbAt(0, 0)[2], 0f);
    }

    @Test
    void rejectsInvalidProcessorConfigurationAndRgbInput() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2VisionImageProcessor(16, 100, 3));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(2, 2, new byte[3]));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionImageProcessor.RgbImage.fromFloatSamples(
                        1, 1, new float[]{0f, Float.NaN, 255f}));
        assertThrows(IllegalArgumentException.class,
                () -> processor().aspectRatioPreservingSize(0, 10));
    }

    private static EmbeddingGemma2VisionImageProcessor processor() {
        return new EmbeddingGemma2VisionImageProcessor(
                PATCH_SIZE, MAX_SOFT_TOKENS, POOLING_KERNEL);
    }
}
