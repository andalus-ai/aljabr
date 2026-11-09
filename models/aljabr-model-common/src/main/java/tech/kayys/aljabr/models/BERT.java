package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.TransformerEncoder;

/**
 * BERT (Bidirectional Encoder Representations from Transformers) model family.
 *
 * <p>Implements the encoder-only Transformer architecture from
 * <em>"BERT: Pre-training of Deep Bidirectional Transformers"</em> (Devlin et al., 2018).
 *
 * <p>Available variants:
 * <ul>
 *   <li>{@link #bertBase(int)} — 12 layers, 768 hidden, 12 heads, ~110M params</li>
 *   <li>{@link #bertLarge(int)} — 24 layers, 1024 hidden, 16 heads, ~340M params</li>
 *   <li>{@link #forSequenceClassification(int, int)} — BERT-base + classification head</li>
 * </ul>
 *
 * <h3>Example — fine-tuning for classification</h3>
 * <pre>{@code
 * NNModule model = BERT.forSequenceClassification(2, 30522, backend);
 * Tensor logits = model.forward(embeddings);
 * }</pre>
 */
public final class BERT {

    /** BERT-base vocabulary size (WordPiece). */
    public static final int VOCAB_SIZE_BASE = 30522;

    /** BERT-large vocabulary size. */
    public static final int VOCAB_SIZE_LARGE = 30522;

    private BERT() {}

    /** @deprecated Use {@link #bertBase(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static TransformerEncoder bertBase(int vocabSize) {
        return bertBase(null, vocabSize);
    }

    /** Constructs BERT-base with all module state owned by {@code backend}. */
    public static TransformerEncoder bertBase(ComputeBackend backend, int vocabSize) {
        TransformerEncoder.Builder builder = backend == null
                ? TransformerEncoder.builder()
                : TransformerEncoder.builder(backend);
        return builder
                .dModel(768)
                .nHeads(12)
                .dFF(3072)
                .nLayers(12)
                .maxSeqLen(512)
                .dropout(0.1f)
                .build();
    }

    /** @deprecated Use {@link #bertLarge(ComputeBackend, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static TransformerEncoder bertLarge(int vocabSize) {
        return bertLarge(null, vocabSize);
    }

    /** Constructs BERT-large with all module state owned by {@code backend}. */
    public static TransformerEncoder bertLarge(ComputeBackend backend, int vocabSize) {
        TransformerEncoder.Builder builder = backend == null
                ? TransformerEncoder.builder()
                : TransformerEncoder.builder(backend);
        return builder
                .dModel(1024)
                .nHeads(16)
                .dFF(4096)
                .nLayers(24)
                .maxSeqLen(512)
                .dropout(0.1f)
                .build();
    }

    /** @deprecated Use {@link #forSequenceClassification(int, int, ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static NNModule forSequenceClassification(int numClasses, int vocabSize) {
        return forSequenceClassification(numClasses, vocabSize, null);
    }

    /** Constructs BERT-base with a classification head on the same explicit backend. */
    public static NNModule forSequenceClassification(
            int numClasses, int vocabSize, ComputeBackend backend) {
        return new BertForClassification(bertBase(backend, vocabSize), numClasses);
    }

    /** BERT encoder with a linear classification head on the [CLS] token. */
    static final class BertForClassification extends NNModule {

        private final TransformerEncoder encoder;
        private final Linear classifier;

        BertForClassification(TransformerEncoder encoder, int numClasses) {
            super(encoder.computeBackend());
            this.encoder = register("encoder", encoder);
            this.classifier = register("classifier", computeBackend() == null
                    ? new Linear(768, numClasses)
                    : new Linear(computeBackend(), 768, numClasses));
        }

        @Override
        public Tensor forward(Tensor x) {
            validateInputBackend(x);
            Tensor encoded = encoder.forward(x);
            long[] shape = encoded.shape().dims();
            int batch = Math.toIntExact(shape[0]);
            int modelDim = Math.toIntExact(shape[2]);
            float[] cls = new float[Math.multiplyExact(batch, modelDim)];
            float[] encodedValues = encoded.toFloatArray();
            for (int b = 0; b < batch; b++) {
                System.arraycopy(encodedValues, b * Math.toIntExact(shape[1]) * modelDim,
                        cls, b * modelDim, modelDim);
            }
            return classifier.forward(tensorOf(cls, batch, modelDim));
        }
    }
}
