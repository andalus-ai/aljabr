package tech.kayys.aljabr.spi.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.Map;
import org.junit.jupiter.api.Test;

class WordPieceTokenizerTest {
    private WordPieceTokenizer tokenizer() {
        return new WordPieceTokenizer(Map.ofEntries(
                Map.entry("[UNK]", 0), Map.entry("[CLS]", 1), Map.entry("[SEP]", 2),
                Map.entry("[PAD]", 3), Map.entry("hello", 4), Map.entry("world", 5),
                Map.entry("un", 6), Map.entry("##aff", 7), Map.entry("play", 8),
                Map.entry("##ing", 9), Map.entry(",", 10), Map.entry("!", 11)),
                Map.of("[CLS]", 1, "[SEP]", 2, "[PAD]", 3), 0, 1, 2, 3,
                "##", true, false, true, 100);
    }

    @Test
    void encodesBasicTokensAndGreedySubwords() {
        assertArrayEquals(new long[] {1, 4, 5, 2},
                tokenizer().encode("Hello world", new EncodeOptions().addBos(true).addEos(true)));
        assertArrayEquals(new long[] {8, 9, 10, 11},
                tokenizer().encode("playing,!", EncodeOptions.defaultOptions()));
    }

    @Test
    void unknownWordIsEmittedAsSingleUnknownToken() {
        assertArrayEquals(new long[] {0}, tokenizer().encode("unseen", EncodeOptions.defaultOptions()));
    }

    @Test
    void decodesContinuationPiecesWithoutSpaces() {
        assertEquals("playing,!", tokenizer().decode(new long[] {8, 9, 10, 11}, DecodeOptions.defaultOptions()));
    }
}
