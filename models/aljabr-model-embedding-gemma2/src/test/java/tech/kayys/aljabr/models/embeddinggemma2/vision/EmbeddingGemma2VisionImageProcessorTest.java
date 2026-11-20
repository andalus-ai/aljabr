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
        assertEquals(-1, positions[279][0]);
        assertEquals(-1, positions[279][1]);
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
        assertEquals(-1, positions[0][219][0]);
        assertEquals(-1, positions[0][220][0]);
        assertEquals(-1, positions[1][219][1]);
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
