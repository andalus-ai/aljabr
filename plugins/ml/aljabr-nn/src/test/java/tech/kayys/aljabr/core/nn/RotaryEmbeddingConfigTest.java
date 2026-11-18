package tech.kayys.aljabr.core.nn;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RotaryEmbeddingConfigTest {

    @Test
    void standardConfigUsesInterleavedAndNoScaling() {
        RotaryEmbeddingConfig config = RotaryEmbeddingConfig.standard(64);
        assertEquals(64, config.rotaryDimension());
        assertEquals(10_000f, config.frequencyBase());
        assertEquals(RotaryEmbeddingConfig.Pairing.INTERLEAVED, config.pairing());
        assertEquals(RotaryEmbeddingConfig.ScalingType.NONE, config.scaling().type());
        assertEquals(1f, config.scaling().factor());
    }

    @Test
    void linearScalingMapsPositionsByFactor() {
        var scaling = RotaryEmbeddingConfig.Scaling.linear(4f);
        assertEquals(2.5f, scaling.scaledPosition(10f));
    }

    @Test
    void validatesRotaryDimensionAndFrequencyBase() {
        assertThrows(IllegalArgumentException.class,
                () -> RotaryEmbeddingConfig.standard(3));
        assertThrows(IllegalArgumentException.class,
                () -> new RotaryEmbeddingConfig(8, 1f,
                        RotaryEmbeddingConfig.Pairing.INTERLEAVED,
                        RotaryEmbeddingConfig.Scaling.none()));
    }

    @Test
    void rejectsInvalidScalingFactor() {
        assertThrows(IllegalArgumentException.class,
                () -> RotaryEmbeddingConfig.Scaling.linear(0.5f));
        assertThrows(IllegalArgumentException.class,
                () -> new RotaryEmbeddingConfig.Scaling(
                        RotaryEmbeddingConfig.ScalingType.NONE, 2f));
    }
}
