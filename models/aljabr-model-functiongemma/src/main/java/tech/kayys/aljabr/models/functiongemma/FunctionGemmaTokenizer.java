package tech.kayys.aljabr.models.functiongemma;

import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;
import tech.kayys.aljabr.spi.tokenizer.TokenizerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * FunctionGemma tokenizer policy layered over Aljabr's checkpoint-loaded
 * tokenizer implementation.
 *
 * <p>Vocabulary, merges, and special-token IDs always come from the model
 * checkpoint's tokenizer.json. This adapter validates the FunctionGemma
 * control-token contract and adds the model-specific generation stop token;
 * it never hard-codes token IDs.</p>
 */
public final class FunctionGemmaTokenizer implements Tokenizer {

    public static final String START_FUNCTION_DECLARATION = "<start_function_declaration>";
    public static final String END_FUNCTION_DECLARATION = "<end_function_declaration>";
    public static final String START_FUNCTION_CALL = "<start_function_call>";
    public static final String END_FUNCTION_CALL = "<end_function_call>";
    public static final String START_FUNCTION_RESPONSE = "<start_function_response>";
    public static final String END_FUNCTION_RESPONSE = "<end_function_response>";
    public static final String ESCAPE = "<escape>";

    private static final Set<String> REQUIRED_SPECIAL_TOKENS = Set.of(
            START_FUNCTION_DECLARATION, END_FUNCTION_DECLARATION,
            START_FUNCTION_CALL, END_FUNCTION_CALL,
            START_FUNCTION_RESPONSE, END_FUNCTION_RESPONSE, ESCAPE);

    private final Tokenizer delegate;
    private final int functionResponseStartId;
    private final Map<String, Integer> specialTokens;

    private FunctionGemmaTokenizer(Tokenizer delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.specialTokens = Map.copyOf(delegate.specialTokens());
        for (String token : REQUIRED_SPECIAL_TOKENS) {
            if (!specialTokens.containsKey(token)) {
                throw new IllegalArgumentException(
                        "FunctionGemma tokenizer.json is missing required special token " + token);
            }
        }
        this.functionResponseStartId = specialTokens.get(START_FUNCTION_RESPONSE);
    }

    /**
     * Loads tokenizer.json from a local Hugging Face model directory and
     * validates the FunctionGemma-specific special-token inventory.
     */
    public static FunctionGemmaTokenizer load(Path modelDirectory) throws IOException {
        return from(TokenizerFactory.load(modelDirectory, null));
    }

    public static FunctionGemmaTokenizer from(Tokenizer tokenizer) {
        return new FunctionGemmaTokenizer(tokenizer);
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        return delegate.encode(text, options);
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        return delegate.decode(tokens, options);
    }

    @Override public int vocabSize() { return delegate.vocabSize(); }
    @Override public int bosTokenId() { return delegate.bosTokenId(); }
    @Override public int eosTokenId() { return delegate.eosTokenId(); }
    @Override public int padTokenId() { return delegate.padTokenId(); }

    @Override
    public int[] allStopTokenIds() {
        Set<Integer> stopIds = new LinkedHashSet<>();
        if (delegate.eosTokenId() >= 0) {
            stopIds.add(delegate.eosTokenId());
        }
        stopIds.add(functionResponseStartId);
        return stopIds.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public Map<String, Integer> specialTokens() {
        return specialTokens;
    }

    public int tokenId(String token) {
        Integer id = specialTokens.get(token);
        if (id == null) {
            throw new IllegalArgumentException("Unknown FunctionGemma control token: " + token);
        }
        return id;
    }
}
