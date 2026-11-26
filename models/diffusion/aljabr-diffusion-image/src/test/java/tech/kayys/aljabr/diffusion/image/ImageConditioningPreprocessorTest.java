package tech.kayys.aljabr.diffusion.image;

import org.junit.jupiter.api.Test;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import static org.junit.jupiter.api.Assertions.*;

class ImageConditioningPreprocessorTest {
    @Test void decodesAndPreservesShapeLayoutAndProvenance() {
        var input = imageInput();
        var adapter = new ImageConditioningPreprocessor(ignored ->
                new DecodedImage(2, 1, 3, DecodedImage.Layout.HWC,
                        new float[]{.1f, .2f, .3f, .4f, .5f, .6f}));
        var result = (ImageConditioningData) adapter.prepare(input, DiffusionConditioningPreparationContext.defaults());
        assertEquals("reference", result.inputId());
        assertEquals(DiffusionConditioningInput.Modality.REFERENCE_IMAGE, result.modality());
        assertEquals(2, result.width());
        assertEquals(1, result.height());
        assertEquals(3, result.channels());
        assertEquals(DecodedImage.Layout.HWC, result.layout());
        assertArrayEquals(new float[]{.1f, .2f, .3f, .4f, .5f, .6f}, result.pixels());
    }

    @Test void delegatesModelSpecificOperationsToInjectedTransform() {
        var input = imageInput();
        var invoked = new AtomicBoolean();
        var context = new DiffusionConditioningPreparationContext(Map.of("image.resize", "512x512"));
        var adapter = new ImageConditioningPreprocessor(
                ignored -> new DecodedImage(1, 1, 1, DecodedImage.Layout.HWC, new float[]{.5f}),
                (actualInput, decoded, actualContext) -> {
                    assertEquals(input, actualInput);
                    assertEquals("512x512", actualContext.options().get("image.resize"));
                    invoked.set(true);
                    return new DecodedImage(2, 2, 1, DecodedImage.Layout.HWC, new float[]{.1f, .2f, .3f, .4f});
                });
        var result = (ImageConditioningData) adapter.prepare(input, context);
        assertTrue(invoked.get());
        assertEquals(2, result.width());
        assertEquals(2, result.height());
    }

    @Test void rejectsUnsupportedModality() {
        var adapter = new ImageConditioningPreprocessor(ignored ->
                new DecodedImage(1, 1, 1, DecodedImage.Layout.HWC, new float[]{0}));
        var prompt = new DiffusionConditioningInput("prompt", DiffusionConditioningInput.Modality.PROMPT, "prompt:one");
        assertFalse(adapter.supports(prompt));
        assertThrows(DiffusionConditioningPreparationException.class,
                () -> adapter.prepare(prompt, DiffusionConditioningPreparationContext.defaults()));
    }

    @Test void wrapsDecoderFailureWithInputProvenance() {
        var adapter = new ImageConditioningPreprocessor(ignored -> { throw new IllegalStateException("cannot decode"); });
        var failure = assertThrows(DiffusionConditioningPreparationException.class,
                () -> adapter.prepare(imageInput(), DiffusionConditioningPreparationContext.defaults()));
        assertEquals("reference", failure.inputId());
        assertEquals(DiffusionConditioningInput.Modality.REFERENCE_IMAGE, failure.modality());
        assertInstanceOf(IllegalStateException.class, failure.getCause());
    }

    @Test void pixelStorageIsDefensivelyCopiedAndValidated() {
        float[] source = {.25f};
        var image = new DecodedImage(1, 1, 1, DecodedImage.Layout.CHW, source);
        source[0] = 9f;
        float[] returned = image.pixels();
        returned[0] = 8f;
        assertArrayEquals(new float[]{.25f}, image.pixels());
        assertThrows(IllegalArgumentException.class,
                () -> new DecodedImage(2, 2, 3, DecodedImage.Layout.HWC, new float[]{1f}));
        assertThrows(IllegalArgumentException.class,
                () -> new DecodedImage(1, 1, 1, DecodedImage.Layout.HWC, new float[]{Float.NaN}));
    }
    private static DiffusionConditioningInput imageInput() {
        return new DiffusionConditioningInput("reference",
                DiffusionConditioningInput.Modality.REFERENCE_IMAGE, "asset:image/reference");
    }
}
