package tech.kayys.aljabr.ml.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.models.Whisper;
import static org.junit.jupiter.api.Assertions.*;

class WhisperTest {

    @Test
    void testWhisperForwardShapes() {
        Whisper.WhisperConfig config = Whisper.WhisperConfig.tiny();
        Whisper model = new Whisper(config);

        // Mel Spectrogram shape: [batch, dMels, timeFrames]
        // Whisper Tiny uses dMels=80, and typically max 3000 frames (30 seconds)
        Tensor mel = Tensor.randn(1, 80, 100);
        
        // 1. Encoder forward
        Tensor audioFeatures = model.encoder.forward(mel);
        
        // Encoder strides by 2, so 100 frames -> 50 sequence length
        // Shape: [batch, seqLen, dModel]
        assertEquals(1, audioFeatures.shape().dim(0));
        assertEquals(50, audioFeatures.shape().dim(1));
        assertEquals(config.dModel, audioFeatures.shape().dim(2));

        // 2. Decoder decode step
        // Tokens shape: [batch, seqLen]
        Tensor tokens = Tensor.of(new float[]{50257, 50259}, 1, 2); // BOS tokens
        Tensor logits = model.decode(tokens, audioFeatures);
        
        // Output shape: [batch, seqLen, vocabSize]
        assertEquals(1, logits.shape().dim(0));
        assertEquals(2, logits.shape().dim(1));
        assertEquals(config.vocabSize, logits.shape().dim(2));
    }
}
