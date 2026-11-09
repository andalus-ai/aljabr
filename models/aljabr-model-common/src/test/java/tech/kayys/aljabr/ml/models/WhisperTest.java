package tech.kayys.aljabr.ml.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.models.Whisper;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WhisperTest {

    @Test
    void testWhisperForwardShapes() {
        CpuBackend backend = new CpuBackend();
        Whisper.WhisperConfig config = Whisper.WhisperConfig.tiny();
        try (Whisper model = new Whisper(backend, config)) {
            Tensor mel = TensorFactory.randn(backend, 1, 80, 100);
            Tensor audioFeatures = null;
            Tensor tokens = null;
            Tensor logits = null;
            try {
                // Mel spectrogram: [batch, dMels, timeFrames].
                audioFeatures = model.encoder.forward(mel);

                assertEquals(1, audioFeatures.shape().dim(0));
                assertEquals(50, audioFeatures.shape().dim(1));
                assertEquals(config.dModel, audioFeatures.shape().dim(2));

                tokens = TensorFactory.of(new float[] {50257, 50259}, backend, 1, 2);
                logits = model.decode(tokens, audioFeatures);

                assertEquals(1, logits.shape().dim(0));
                assertEquals(2, logits.shape().dim(1));
                assertEquals(config.vocabSize, logits.shape().dim(2));
            } finally {
                if (logits != null) logits.release();
                if (tokens != null) tokens.release();
                if (audioFeatures != null) audioFeatures.release();
                mel.release();
            }
        } finally {
            backend.close();
        }
    }
}
