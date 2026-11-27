package tech.kayys.aljabr.safetensor.bert;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.bert.MiniLmL6V2;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

class MiniLmL6V2BundleContractTest {

    @Test
    void acceptsDistinctSpecialTokensInsideOfficialVocabulary() {
        assertDoesNotThrow(() -> MiniLmL6V2Bundle.validateTokenizerContract(
                MiniLmL6V2.config(), tokenizer(101, 102, 0)));
    }

    @Test
    void rejectsSpecialTokenIdAtVocabularyBoundary() {
        assertThrows(IllegalArgumentException.class, () ->
                MiniLmL6V2Bundle.validateTokenizerContract(
                        MiniLmL6V2.config(), tokenizer(101, 102, 30_522)));
    }

    @Test
    void rejectsNegativeSpecialTokenId() {
        assertThrows(IllegalArgumentException.class, () ->
                MiniLmL6V2Bundle.validateTokenizerContract(
                        MiniLmL6V2.config(), tokenizer(101, -1, 0)));
    }

    @Test
    void rejectsAliasedSpecialTokenIds() {
        assertThrows(IllegalArgumentException.class, () ->
                MiniLmL6V2Bundle.validateTokenizerContract(
                        MiniLmL6V2.config(), tokenizer(101, 102, 102)));
    }

    private static Tokenizer tokenizer(int bos, int eos, int pad) {
        return new Tokenizer() {
            @Override public long[] encode(String text, EncodeOptions options) { return new long[0]; }
            @Override public String decode(long[] tokens, DecodeOptions options) { return ""; }
            @Override public int vocabSize() { return 30_522; }
            @Override public int bosTokenId() { return bos; }
            @Override public int eosTokenId() { return eos; }
            @Override public int padTokenId() { return pad; }
            @Override public int[] allStopTokenIds() { return new int[] {eos}; }
        };
    }
}
