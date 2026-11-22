package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioRelativePositionEncodingTest {
    @Test
    void producesDescendingSinusoidalPositionsInSinThenCosLayout() {
        var config = new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 1, 1, 8, 3, List.of(2, 2),
                2, 3, 0, 1e-6, 0.5, "silu", "float32", false,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10, 50.0, -1.0e9);

        var encoding = new EmbeddingGemma2AudioRelativePositionEncoding(config);
        float[][] positions = encoding.compute();

        assertEquals(3, positions.length);
        assertEquals(4, encoding.contextSize());
        assertArrayEquals(new float[]{(float) Math.sin(2), (float) Math.cos(2)},
                positions[0], 1e-6f);
        assertArrayEquals(new float[]{(float) Math.sin(1), (float) Math.cos(1)},
                positions[1], 1e-6f);
        assertArrayEquals(new float[]{0f, 1f}, positions[2], 1e-6f);
    }
}
