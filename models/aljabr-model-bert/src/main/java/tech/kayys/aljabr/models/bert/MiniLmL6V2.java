package tech.kayys.aljabr.models.bert;

import java.util.Objects;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Model profile for sentence-transformers/all-MiniLM-L6-v2.
 *
 * <p>This is a thin model-specific contract over the reusable BERT encoder
 * and sentence-embedding pipeline. Checkpoint loading remains the
 * responsibility of a format adapter such as the Safetensors module.</p>
 */
public final class MiniLmL6V2 {
    public static final String MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2";
    public static final int EMBEDDING_DIMENSION = 384;
    public static final int MAX_SEQUENCE_LENGTH = 256;

    private final BertEncoder encoder;
    private final BertSentenceEmbedder embedder;

    public MiniLmL6V2(BertEncoder encoder) {
        this.encoder = Objects.requireNonNull(encoder, "encoder must not be null");
        validateCompatible(encoder.config());
        this.encoder.eval();
        this.embedder = new BertSentenceEmbedder(encoder);
    }

    /** Creates an encoder with the official model architecture; weights must still be loaded. */
    public static MiniLmL6V2 create(ComputeBackend backend) {
        return new MiniLmL6V2(new BertEncoder(
                Objects.requireNonNull(backend, "backend must not be null"), config()));
    }

    /** Official Hugging Face BERT configuration for this checkpoint. */
    public static BertConfig config() {
        return new BertConfig(
                30_522, 384, 6, 12, 1_536, 512, 2, 0, 1.0e-12, "gelu");
    }

    /** Ensures the supplied generic BERT encoder matches the MiniLM-L6-v2 architecture. */
    public static void validateCompatible(BertConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        BertConfig expected = config();
        if (!expected.equals(config)) {
            throw new IllegalArgumentException("BERT config is not compatible with " + MODEL_ID
                    + "; expected " + expected + " but received " + config);
        }
    }

    /** Returns normalized 384-dimensional sentence embeddings for a padded token batch. */
    public float[][] embed(Tensor inputIds, Tensor attentionMask) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        if (inputIds.shape().rank() != 2 || inputIds.shape().dim(1) > MAX_SEQUENCE_LENGTH) {
            throw new IllegalArgumentException("MiniLM-L6-v2 input sequence length must be between 1 and "
                    + MAX_SEQUENCE_LENGTH);
        }
        return embedder.embed(inputIds, attentionMask);
    }

    public BertEncoder encoder() {
        return encoder;
    }

    public int embeddingDimension() {
        return EMBEDDING_DIMENSION;
    }

    public int maxSequenceLength() {
        return MAX_SEQUENCE_LENGTH;
    }
}
