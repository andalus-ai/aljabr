package tech.kayys.aljabr.safetensor.bert;

import java.util.List;
import java.util.Objects;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.models.bert.MiniLmL6V2;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

/**
 * Fully loaded MiniLM-L6-v2 bundle: model weights plus its Hugging Face tokenizer.
 * The compute backend remains caller-owned; closing this bundle closes only
 * the model's registered tensor/module state.
 */
public final class MiniLmL6V2Bundle implements AutoCloseable {
    private final MiniLmL6V2 model;
    private final Tokenizer tokenizer;

    public MiniLmL6V2Bundle(MiniLmL6V2 model, Tokenizer tokenizer) {
        this.model = Objects.requireNonNull(model, "model must not be null");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer must not be null");
        int vocabularySize = tokenizer.vocabSize();
        if (vocabularySize != model.encoder().config().vocabularySize()) {
            throw new IllegalArgumentException("Tokenizer vocabulary size " + vocabularySize
                    + " does not match BERT vocabulary size "
                    + model.encoder().config().vocabularySize());
        }
        validateSpecialToken("CLS/BOS", tokenizer.bosTokenId(), vocabularySize);
        validateSpecialToken("SEP/EOS", tokenizer.eosTokenId(), vocabularySize);
        validateSpecialToken("PAD", tokenizer.padTokenId(), vocabularySize);
        if (tokenizer.bosTokenId() == tokenizer.eosTokenId()
                || tokenizer.bosTokenId() == tokenizer.padTokenId()
                || tokenizer.eosTokenId() == tokenizer.padTokenId()) {
            throw new IllegalArgumentException("MiniLM CLS, SEP and PAD token IDs must be distinct");
        }
    }

    private static void validateSpecialToken(String name, int tokenId, int vocabularySize) {
        if (tokenId < 0 || tokenId >= vocabularySize) {
            throw new IllegalArgumentException("MiniLM tokenizer " + name + " token ID " + tokenId
                    + " is outside vocabulary range [0, " + vocabularySize + ")");
        }
    }

    /** Tokenizes, truncates, pads, runs the encoder and returns normalized 384-D vectors. */
    public float[][] encode(List<String> sentences) {
        Objects.requireNonNull(sentences, "sentences must not be null");
        if (sentences.isEmpty()) return new float[0][];
        int batch = sentences.size();
        long[][] encoded = new long[batch][];
        int sequence = 1;
        for (int row = 0; row < batch; row++) {
            String sentence = Objects.requireNonNull(sentences.get(row),
                    "sentence at index " + row + " must not be null");
            long[] ids = tokenizer.encode(sentence, new EncodeOptions().addBos(true).addEos(true));
            if (ids.length == 0) ids = new long[] {tokenizer.bosTokenId(), tokenizer.eosTokenId()};
            if (ids.length > MiniLmL6V2.MAX_SEQUENCE_LENGTH) {
                ids = java.util.Arrays.copyOf(ids, MiniLmL6V2.MAX_SEQUENCE_LENGTH);
                ids[ids.length - 1] = tokenizer.eosTokenId();
            }
            encoded[row] = ids;
            sequence = Math.max(sequence, ids.length);
        }

        int elementCount = Math.multiplyExact(batch, sequence);
        int[] inputIds = new int[elementCount];
        int[] attentionMask = new int[elementCount];
        java.util.Arrays.fill(inputIds, tokenizer.padTokenId());
        for (int row = 0; row < batch; row++) {
            long[] tokens = encoded[row];
            for (int column = 0; column < tokens.length; column++) {
                long token = tokens[column];
                if (token < 0 || token >= tokenizer.vocabSize() || token > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("Tokenizer emitted out-of-range token ID " + token);
                }
                int offset = row * sequence + column;
                inputIds[offset] = (int) token;
                attentionMask[offset] = 1;
            }
        }

        Tensor idsTensor = TensorFactory.of(inputIds, model.encoder().computeBackend(), batch, sequence);
        Tensor maskTensor = null;
        try {
            maskTensor = TensorFactory.of(attentionMask, model.encoder().computeBackend(), batch, sequence);
            return model.embed(idsTensor, maskTensor);
        } finally {
            if (maskTensor != null) maskTensor.release();
            idsTensor.release();
        }
    }

    public MiniLmL6V2 model() {
        return model;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public void close() {
        model.encoder().close();
    }
}
