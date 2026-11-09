package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
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

    /** @deprecated Supply the compute backend selected by the application/runtime. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Whisper(WhisperConfig config) {
        this(null, config);
    }

    public Whisper(ComputeBackend backend, WhisperConfig config) {
        super(backend);
        this.encoder = registerModule("encoder", new WhisperEncoder(backend, config));
        this.decoder = registerModule("decoder", new WhisperDecoder(backend, config));
    }

    /** Standard forwarding passes encoder over audio, returning audio features. */
    @Override
    public Tensor forward(Tensor melSpectrogram) {
        validateInputBackend(melSpectrogram);
        return encoder.forward(melSpectrogram);
    }

    /** Decode step taking hidden audio features and previously generated tokens. */
    public Tensor decode(Tensor tokens, Tensor audioFeatures) {
        validateInputBackend(tokens);
        validateInputBackend(audioFeatures);
        return decoder.forward(tokens, audioFeatures);
    }

    public static class WhisperEncoder extends NNModule {
        private final Conv1d conv1;
        private final Conv1d conv2;
        private final Sequential layers;
        private final Tensor positionEmbeddings;
        private final GELU activation;

        /** @deprecated Supply an explicit compute backend. */
        @Deprecated(since = "0.2.0", forRemoval = true)
        public WhisperEncoder(WhisperConfig config) {
            this(null, config);
        }

        public WhisperEncoder(ComputeBackend backend, WhisperConfig config) {
            super(backend);
            this.conv1 = registerModule("conv1", new Conv1d(backend, config.dMels, config.dModel, 3, 1, 1));
            this.conv2 = registerModule("conv2", new Conv1d(backend, config.dModel, config.dModel, 3, 2, 1));
            this.positionEmbeddings = register("positionEmbeddings",
                    zeros(1, config.maxSourcePos, config.dModel));
            this.activation = registerModule("activation", new GELU());

            this.layers = new Sequential(backend);
            for (int i = 0; i < config.encoderLayers; i++) {
                this.layers.add(String.valueOf(i), new TransformerEncoderLayer(
                        backend, config.dModel, config.encoderHeads, config.dModel * 4, 0.1f));
            }
            registerModule("layers", layers);
        }

        @Override
        public Tensor forward(Tensor x) {
            validateInputBackend(x);
            x = activation.forward(conv1.forward(x));
            x = activation.forward(conv2.forward(x));

            // [batch, dModel, seqLen] -> [batch, seqLen, dModel]
            Tensor transposed = x.transpose();
            int seqLen = Math.toIntExact(transposed.shape().dim(1));
            Tensor embedded = transposed.add(cropPositionEmbeddings(
                    computeBackend(), positionEmbeddings, seqLen));
            return layers.forward(embedded);
        }
    }

    public static class WhisperDecoder extends NNModule {
        private final Embedding tokenEmbedding;
        private final Tensor positionEmbeddings;
        private final Sequential layers;
        private final Linear outputProjection;

        /** @deprecated Supply an explicit compute backend. */
        @Deprecated(since = "0.2.0", forRemoval = true)
        public WhisperDecoder(WhisperConfig config) {
            this(null, config);
        }

        public WhisperDecoder(ComputeBackend backend, WhisperConfig config) {
            super(backend);
            this.tokenEmbedding = registerModule("tokenEmbedding",
                    new Embedding(backend, config.vocabSize, config.dModel));
            this.positionEmbeddings = register("positionEmbeddings",
                    zeros(1, config.maxTargetPos, config.dModel));

            this.layers = new Sequential(backend);
            for (int i = 0; i < config.decoderLayers; i++) {
                this.layers.add(String.valueOf(i), new TransformerDecoderLayer(
                        backend, config.dModel, config.decoderHeads, config.dModel * 4, 0.1f));
            }
            registerModule("layers", layers);

            this.outputProjection = registerModule("outputProjection",
                    new Linear(backend, config.dModel, config.vocabSize));
        }

        public Tensor forward(Tensor tokens, Tensor audioFeatures) {
            validateInputBackend(tokens);
            validateInputBackend(audioFeatures);
            Tensor x = tokenEmbedding.forward(tokens);
            int seqLen = Math.toIntExact(x.shape().dim(1));
            x = x.add(cropPositionEmbeddings(computeBackend(), positionEmbeddings, seqLen));
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
            throw new UnsupportedOperationException(
                    "WhisperDecoder requires memory/audioFeatures context. Use decode().");
        }
    }

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

    private static Tensor cropPositionEmbeddings(
            ComputeBackend backend, Tensor positionEmbeddings, int seqLen) {
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
        return TensorFactory.of(cropped, backend, 1, clampedSeqLen, dModel);
    }
}
