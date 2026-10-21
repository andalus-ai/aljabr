package tech.kayys.aljabr.models;

import tech.kayys.tafkir.ml.audio.AudioBuffer;
import tech.kayys.tafkir.ml.audio.MelSpectrogram;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import java.util.ArrayList;
import java.util.List;

/**
 * High-level orchestration for Whisper STT (Speech-to-Text).
 * <p>
 * Handles the full pipeline:
 * 1. Audio Preprocessing (SIMD-accelerated via AudioBuffer/MelSpectrogram)
 * 2. Encoder Pass (Audio Feature Extraction)
 * 3. Decoder Loop (Greedy Decoding for Transcription)
 */
public class WhisperPipeline {

    private final Whisper model;
    private final MelSpectrogram processor;
    private final Whisper.WhisperConfig config;

    public WhisperPipeline(Whisper model, MelSpectrogram processor, Whisper.WhisperConfig config) {
        this.model = model;
        this.processor = processor;
        this.config = config;
    }

    /**
     * Transcribes raw audio from an AudioBuffer.
     * Uses Greedy Search to generate token IDs.
     * 
     * @param audio the input audio buffer (must be 16kHz for standard Whisper)
     * @return a list of generated token IDs
     */
    public List<Integer> transcribe(AudioBuffer audio) {
        // 1. Audio to Mel Spectrogram
        // Convert AudioBuffer to Tensor [1, d_mels, time]
        float[] audioData = audio.toArray();
        Tensor audioTensor = Tensor.of(audioData, 1, audioData.length);
        Tensor mel = processor.forward(audioTensor);

        // 2. Encoder Pass
        Tensor audioFeatures = model.encoder.forward(mel);

        // 3. Greedy Decoding Loop
        List<Integer> tokens = new ArrayList<>();
        tokens.add(config.sot);
        tokens.add(config.langEn);
        tokens.add(config.transcribe);
        tokens.add(config.noTimestamps);

        int maxNewTokens = config.maxTargetPos;
        for (int i = 0; i < maxNewTokens; i++) {
            // Prepare decoder input: [batch=1, seq_len]
            float[] tokenData = new float[tokens.size()];
            for (int t = 0; t < tokens.size(); t++) tokenData[t] = tokens.get(t);
            Tensor decoderIn = Tensor.of(tokenData, 1, tokens.size());

            // Decoder forward: [1, seq_len, vocab_size]
            Tensor logits = model.decoder.forward(decoderIn, audioFeatures);
            
            int nextTokenId = selectNextTokenId(logits, tokens.size() - 1);
            
            System.out.print("."); // Progress indicator
            System.out.flush();

            if (nextTokenId == config.eot) {
                break;
            }

            tokens.add(nextTokenId);
        }
        System.out.println(); // New line after dots

        return tokens;
    }

    /**
     * The Tensor compatibility surface does not currently expose full
     * indexing helpers, so we scan the final-step logits directly.
     */
    private static int selectNextTokenId(Tensor logits, int sequenceIndex) {
        long[] shape = logits.shape().dims();
        if (shape.length != 3 || shape[0] != 1) {
            throw new IllegalArgumentException("Expected logits shape [1, seq, vocab], got "
                    + java.util.Arrays.toString(shape));
        }

        int seqLen = Math.toIntExact(shape[1]);
        int vocabSize = Math.toIntExact(shape[2]);
        if (sequenceIndex < 0 || sequenceIndex >= seqLen) {
            throw new IllegalArgumentException("sequenceIndex out of bounds: " + sequenceIndex);
        }

        float[] values = logits.toFloatArray();
        int base = sequenceIndex * vocabSize;
        int bestToken = 0;
        float bestScore = values[base];

        for (int token = 1; token < vocabSize; token++) {
            float score = values[base + token];
            if (score > bestScore) {
                bestScore = score;
                bestToken = token;
            }
        }
        return bestToken;
    }
}
