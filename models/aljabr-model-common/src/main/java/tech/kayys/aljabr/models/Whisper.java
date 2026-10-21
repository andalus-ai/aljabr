package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.Sequential;
import tech.kayys.aljabr.core.nn.Embedding;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.GELU;
import tech.kayys.aljabr.core.nn.Conv1d;
import tech.kayys.aljabr.core.nn.TransformerEncoderLayer;
import tech.kayys.aljabr.core.nn.TransformerDecoderLayer;

/**
 * Whisper Model Architecture natively in Java.
 * Includes Convolutional stem, Transformer Encoder (for Audio), 
 * and Transformer Decoder (for Text generation).
 */
public class Whisper extends NNModule {

    public final WhisperEncoder encoder;
    public final WhisperDecoder decoder;

    public Whisper(WhisperConfig config) {
        this.encoder = new WhisperEncoder(config);
        this.decoder = new WhisperDecoder(config);
    }

    /**
     * Standard forwarding passes encoder over audio, returning audio features.
     */
    @Override
    public Tensor forward(Tensor melSpectrogram) {
        return encoder.forward(melSpectrogram);
    }

    /**
     * Decode step taking hidden audio features and previously generated tokens.
     */
    public Tensor decode(Tensor tokens, Tensor audioFeatures) {
        return decoder.forward(tokens, audioFeatures);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Sub-Modules
    // ─────────────────────────────────────────────────────────────────────────

    public static class WhisperEncoder extends NNModule {
        private final Conv1d conv1;
        private final Conv1d conv2;
        private final Sequential layers;
        private final Tensor positionEmbeddings;
        private final GELU activation = new GELU();
        
        public WhisperEncoder(WhisperConfig config) {
            // Audio stem (e.g. kernel 3, stride 1 & 2)
            this.conv1 = new Conv1d(config.dMels, config.dModel, 3, 1, 1);
            this.conv2 = new Conv1d(config.dModel, config.dModel, 3, 2, 1); // Stride 2 halves length
            
            this.positionEmbeddings = register("positionEmbeddings", Tensor.zeros(1, config.maxSourcePos, config.dModel));
            
            this.layers = new Sequential();
            for (int i = 0; i < config.encoderLayers; i++) {
                this.layers.add(new TransformerEncoderLayer(
                    config.dModel, config.encoderHeads, config.dModel * 4, 0.1f
                ));
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            // x shape: [batch, dMels, timeFrames]
            x = activation.forward(conv1.forward(x));
            x = activation.forward(conv2.forward(x));
            
            // Expected sequence shape for transformer typically [batch, seqLen, dModel]
            // We'd transpose x: [batch, dModel, seqLen] -> [batch, seqLen, dModel]
            Tensor transposed = x.transpose();
            
            // Add Positional Embeddings
            int seqLen = Math.toIntExact(transposed.shape().dim(1));
            Tensor embedded = transposed.add(cropPositionEmbeddings(positionEmbeddings, seqLen));

            return layers.forward(embedded);
        }
    }

    public static class WhisperDecoder extends NNModule {
        private final Embedding tokenEmbedding;
        private final Tensor positionEmbeddings;
        private final Sequential layers;
        private final Linear outputProjection;

        public WhisperDecoder(WhisperConfig config) {
            this.tokenEmbedding = new Embedding(config.vocabSize, config.dModel);
            this.positionEmbeddings = register("positionEmbeddings", Tensor.zeros(1, config.maxTargetPos, config.dModel));
            
            this.layers = new Sequential();
            for (int i = 0; i < config.decoderLayers; i++) {
                // Cross-attention implies DecoderLayer
                this.layers.add(new TransformerDecoderLayer(
                    config.dModel, config.decoderHeads, config.dModel * 4, 0.1f
                ));
            }
            
            this.outputProjection = new Linear(config.dModel, config.vocabSize);
        }

        public Tensor forward(Tensor tokens, Tensor audioFeatures) {
            Tensor x = tokenEmbedding.forward(tokens);
            // Add PE
            int seqLen = Math.toIntExact(x.shape().dim(1));
            x = x.add(cropPositionEmbeddings(positionEmbeddings, seqLen));

            // Sequence forward - passing audioFeatures to cross-attention
            for (int i = 0; i < layers.size(); i++) {
                NNModule layer = layers.get(i);
                if (layer instanceof TransformerDecoderLayer decoderLayer) {
                    x = decoderLayer.forward(x, audioFeatures);
                } else {
                    x = layer.forward(x);
                }
            }
            
            return outputProjection.forward(x);
        }

        @Override
        public Tensor forward(Tensor input) {
            throw new UnsupportedOperationException("WhisperDecoder requires memory/audioFeatures context. Use decode().");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration
    // ─────────────────────────────────────────────────────────────────────────

    public static class WhisperConfig {
        public int dMels = 80;
        public int dModel = 512;
        public int vocabSize = 51865;
        public int maxSourcePos = 1500;
        public int maxTargetPos = 448;
        public int encoderLayers = 6;
        public int encoderHeads = 8;
        public int decoderLayers = 6;
        public int decoderHeads = 8;
        
        // Special Tokens (Standard Whisper defaults)
        public int sot = 50258;
        public int eot = 50257;
        public int langEn = 50259;
        public int transcribe = 50359;
        public int noTimestamps = 50363;
        
        public static WhisperConfig tiny() {
            WhisperConfig c = new WhisperConfig();
            c.dModel = 384;
            c.encoderLayers = 4;
            c.decoderLayers = 4;
            c.encoderHeads = 6;
            c.decoderHeads = 6;
            return c;
        }
        
        public static WhisperConfig base() {
            return new WhisperConfig();
        }
    }

    private static Tensor cropPositionEmbeddings(Tensor positionEmbeddings, int seqLen) {
        long[] shape = positionEmbeddings.shape().dims();
        if (shape.length != 3 || shape[0] != 1) {
            throw new IllegalArgumentException("Expected positional embeddings shape [1, seq, dModel], got "
                    + java.util.Arrays.toString(shape));
        }

        int availableSeqLen = Math.toIntExact(shape[1]);
        int dModel = Math.toIntExact(shape[2]);
        int clampedSeqLen = Math.min(seqLen, availableSeqLen);

        float[] source = positionEmbeddings.toFloatArray();
        float[] cropped = new float[clampedSeqLen * dModel];
        System.arraycopy(source, 0, cropped, 0, cropped.length);
        return Tensor.of(cropped, 1, clampedSeqLen, dModel);
    }
}
