package tech.kayys.aljabr.models.embeddinggemma2.input;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

/**
 * Adapts Aljabr's shared tokenizer SPI to the EmbeddingGemma 2 text encoder.
 *
 * <p>Task formatting remains model-specific and happens before tokenization.
 * This adapter adds the Gemma BOS token, does not append EOS, and produces
 * the unpadded mask and positions expected by the bidirectional encoder.</p>
 */
public final class EmbeddingGemma2Tokenizer {
    private final EmbeddingGemma2Config config;
    private final Tokenizer tokenizer;

    public EmbeddingGemma2Tokenizer(EmbeddingGemma2Config config, Tokenizer tokenizer) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer must not be null");
        if (tokenizer.vocabSize() < config.vocabSize()) {
            throw new IllegalArgumentException("tokenizer vocabulary (" + tokenizer.vocabSize()
                    + ") is smaller than model vocabulary (" + config.vocabSize() + ")");
        }
        int bos = tokenizer.bosTokenId();
        if (bos < 0 || bos >= config.vocabSize()) {
            throw new IllegalArgumentException("tokenizer must expose a valid Gemma BOS token ID");
        }
    }

    /** Tokenizes text that has already been formatted for its embedding task. */
    public TokenizedInput tokenize(String formattedText) {
        Objects.requireNonNull(formattedText, "formattedText must not be null");
        long[] encoded = tokenizer.encode(formattedText, EncodeOptions.defaultOptions());
        if (encoded == null) {
            throw new IllegalStateException("tokenizer returned null token IDs");
        }
        if (encoded.length + 1 > config.maxInputTokens()) {
            throw new IllegalArgumentException("tokenized input exceeds max_input_tokens: "
                    + config.maxInputTokens());
        }
        if (encoded.length + 1 > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("tokenized input exceeds max_position_embeddings: "
                    + config.maxPositionEmbeddings());
        }

        int[] inputIds = new int[encoded.length + 1];
        inputIds[0] = tokenizer.bosTokenId();
        for (int index = 0; index < encoded.length; index++) {
            long tokenId = encoded[index];
            if (tokenId < 0 || tokenId >= config.vocabSize()) {
                throw new IllegalArgumentException("token ID out of model vocabulary range at index "
                        + index + ": " + tokenId);
            }
            inputIds[index + 1] = Math.toIntExact(tokenId);
        }

        boolean[] validTokens = new boolean[inputIds.length];
        int[] positionIds = new int[inputIds.length];
        for (int index = 0; index < inputIds.length; index++) {
            validTokens[index] = true;
            positionIds[index] = index;
        }
        return new TokenizedInput(inputIds, validTokens, positionIds);
    }

    /** Applies the model's task/role template before tokenizing. */
    public TokenizedInput tokenize(
            String input, String title, EmbeddingTask task, EmbeddingGemma2InputRole role) {
        return tokenize(EmbeddingGemma2InputFormatter.format(input, title, task, role));
    }

    /** Decodes token IDs through the same shared tokenizer implementation. */
    public String decode(int[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        long[] ids = new long[inputIds.length];
        for (int index = 0; index < inputIds.length; index++) {
            if (inputIds[index] < 0 || inputIds[index] >= config.vocabSize()) {
                throw new IllegalArgumentException("token ID out of model vocabulary range at index "
                        + index + ": " + inputIds[index]);
            }
            ids[index] = inputIds[index];
        }
        return tokenizer.decode(ids, DecodeOptions.defaultOptions());
    }

    /** Defensive, unpadded single-sequence inputs for the text backbone. */
    public record TokenizedInput(int[] inputIds, boolean[] validTokens, int[] positionIds) {
        public TokenizedInput {
            inputIds = Objects.requireNonNull(inputIds, "inputIds must not be null").clone();
            validTokens = Objects.requireNonNull(validTokens, "validTokens must not be null").clone();
            positionIds = Objects.requireNonNull(positionIds, "positionIds must not be null").clone();
            if (inputIds.length == 0 || validTokens.length != inputIds.length
                    || positionIds.length != inputIds.length) {
                throw new IllegalArgumentException("token IDs, mask, and positions must have equal non-zero lengths");
            }
        }

        @Override public int[] inputIds() { return inputIds.clone(); }
        @Override public boolean[] validTokens() { return validTokens.clone(); }
        @Override public int[] positionIds() { return positionIds.clone(); }
    }
}
