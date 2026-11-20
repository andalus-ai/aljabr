package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaTokenizerTest {

    @Test
    void exposesCheckpointSpecialIdsAndAddsFunctionResponseStopId() {
        Map<String, Integer> specials = requiredTokens();
        specials.put("<eos>", 2);

        FunctionGemmaTokenizer tokenizer = FunctionGemmaTokenizer.from(new StubTokenizer(specials));

        assertEquals(17, tokenizer.tokenId(FunctionGemmaTokenizer.START_FUNCTION_RESPONSE));
        assertArrayEquals(new int[] {2, 17}, tokenizer.allStopTokenIds());
        assertEquals(specials, tokenizer.specialTokens());
    }

    @Test
    void failsClosedWhenCheckpointOmitsFunctionControlToken() {
        Map<String, Integer> specials = requiredTokens();
        specials.remove(FunctionGemmaTokenizer.ESCAPE);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> FunctionGemmaTokenizer.from(new StubTokenizer(specials)));

        assertTrue(error.getMessage().contains(FunctionGemmaTokenizer.ESCAPE));
    }

    private static Map<String, Integer> requiredTokens() {
        Map<String, Integer> tokens = new HashMap<>();
        tokens.put(FunctionGemmaTokenizer.START_FUNCTION_DECLARATION, 11);
        tokens.put(FunctionGemmaTokenizer.END_FUNCTION_DECLARATION, 12);
        tokens.put(FunctionGemmaTokenizer.START_FUNCTION_CALL, 13);
        tokens.put(FunctionGemmaTokenizer.END_FUNCTION_CALL, 14);
        tokens.put(FunctionGemmaTokenizer.START_FUNCTION_RESPONSE, 17);
        tokens.put(FunctionGemmaTokenizer.END_FUNCTION_RESPONSE, 18);
        tokens.put(FunctionGemmaTokenizer.ESCAPE, 19);
        return tokens;
    }

    private record StubTokenizer(Map<String, Integer> specialTokens) implements Tokenizer {
        @Override public long[] encode(String text, EncodeOptions options) { return new long[0]; }
        @Override public String decode(long[] tokens, DecodeOptions options) { return ""; }
        @Override public int vocabSize() { return 20; }
        @Override public int bosTokenId() { return 1; }
        @Override public int eosTokenId() { return specialTokens.getOrDefault("<eos>", -1); }
        @Override public int padTokenId() { return 0; }
        @Override public int[] allStopTokenIds() { return eosTokenId() < 0 ? new int[0] : new int[] {eosTokenId()}; }
    }
}
