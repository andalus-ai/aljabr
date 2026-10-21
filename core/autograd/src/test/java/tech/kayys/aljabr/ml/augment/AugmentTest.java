package tech.kayys.aljabr.augment;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.ml.autograd.GradTensor;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class AugmentTest {

    @Test
    void testRandomHorizontalFlipCreation() {
        var aug = Augmentation.RandomHorizontalFlip.of(0.5);
        assertNotNull(aug);
    }

    @Test
    void testAugmentationPipeline() {
        var pipeline = new AugmentationPipeline();
        assertNotNull(pipeline);
    }

    @Test
    void testFlipAugmentation() {
        var aug = Augmentation.RandomHorizontalFlip.of(1.0);
        var input = GradTensor.randn(3, 32, 32);
        var rng = new Random(42);
        var output = aug.apply(input, rng);
        assertEquals(input.numel(), output.numel());
    }
}
