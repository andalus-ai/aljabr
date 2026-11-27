package tech.kayys.aljabr.diffusion.mask;

import org.junit.jupiter.api.Test;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import static org.junit.jupiter.api.Assertions.*;

class MaskConditioningPreprocessorTest {
    @Test void binaryMaskAppliesConfiguredThresholdAndPreservesProvenance() {
        var adapter = new MaskConditioningPreprocessor(
                ignored -> new DecodedMask(3, 1, new float[]{0.2f, 0.5f, 0.9f}),
                MaskConditioningOptions.binary(0.5f));
        var result = (MaskConditioningData) adapter.prepare(maskInput(),
                DiffusionConditioningPreparationContext.defaults());
        assertEquals("inpaint-mask", result.inputId());
        assertEquals(DiffusionConditioningInput.Modality.MASK, result.modality());
        assertEquals(MaskSemantics.BINARY, result.semantics());
        assertArrayEquals(new float[]{0f, 1f, 1f}, result.values());
    }

    @Test void softMaskPreservesContinuousWeights() {
        var adapter = new MaskConditioningPreprocessor(
                ignored -> new DecodedMask(2, 1, new float[]{0.25f, 0.75f}),
                MaskConditioningOptions.soft());
        var result = (MaskConditioningData) adapter.prepare(maskInput(),
                DiffusionConditioningPreparationContext.defaults());
        assertEquals(MaskSemantics.SOFT, result.semantics());
        assertArrayEquals(new float[]{0.25f, 0.75f}, result.values());
    }

    @Test void transformReceivesExplicitResamplingPolicyAndContext() {
        var invoked = new AtomicBoolean();
        var context = new DiffusionConditioningPreparationContext(Map.of("mask.resize", "256x256"));
        var options = MaskConditioningOptions.binary(0.4f);
        var adapter = new MaskConditioningPreprocessor(
                ignored -> new DecodedMask(1, 1, new float[]{0.5f}),
                (input, decoded, actualOptions, actualContext) -> {
                    assertEquals(options, actualOptions);
                    assertEquals("256x256", actualContext.options().get("mask.resize"));
                    invoked.set(true);
                    return decoded;
                }, options);
        adapter.prepare(maskInput(), context);
        assertTrue(invoked.get());
    }

    @Test void rejectsBilinearResamplingForBinaryMasks() {
        assertThrows(IllegalArgumentException.class, () ->
                new MaskConditioningOptions(MaskSemantics.BINARY, MaskResampling.BILINEAR, 0.5f));
    }

    @Test void rejectsOutOfRangeValuesAndUnsupportedModalities() {
        assertThrows(IllegalArgumentException.class, () -> new DecodedMask(1, 1, new float[]{1.1f}));
        var adapter = new MaskConditioningPreprocessor(
                ignored -> new DecodedMask(1, 1, new float[]{0f}), MaskConditioningOptions.soft());
        var image = new DiffusionConditioningInput("image", DiffusionConditioningInput.Modality.IMAGE, "asset:image");
        assertFalse(adapter.supports(image));
        assertThrows(DiffusionConditioningPreparationException.class,
                () -> adapter.prepare(image, DiffusionConditioningPreparationContext.defaults()));
    }

    @Test void wrapsDecoderFailureWithInputProvenance() {
        var adapter = new MaskConditioningPreprocessor(
                ignored -> { throw new IllegalStateException("bad mask"); }, MaskConditioningOptions.soft());
        var failure = assertThrows(DiffusionConditioningPreparationException.class,
                () -> adapter.prepare(maskInput(), DiffusionConditioningPreparationContext.defaults()));
        assertEquals("inpaint-mask", failure.inputId());
        assertEquals(DiffusionConditioningInput.Modality.MASK, failure.modality());
        assertInstanceOf(IllegalStateException.class, failure.getCause());
    }

    @Test void maskPixelsAreDefensivelyCopied() {
        float[] source = {0.1f, 0.9f};
        var mask = new DecodedMask(2, 1, source);
        source[0] = 1f;
        float[] returned = mask.values();
        returned[1] = 0f;
        assertArrayEquals(new float[]{0.1f, 0.9f}, mask.values());
    }

    private static DiffusionConditioningInput maskInput() {
        return new DiffusionConditioningInput("inpaint-mask", DiffusionConditioningInput.Modality.MASK, "asset:mask");
    }
}
