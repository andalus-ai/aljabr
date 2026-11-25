package tech.kayys.aljabr.spi.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;
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

    @Test
    void preservesAddedSpecialTokensAndHonorsDecodeOptions() {
        WordPieceTokenizer tokenizer = tokenizer();
        assertArrayEquals(new long[] {1, 4, 2},
                tokenizer.encode("[CLS] Hello [SEP]", EncodeOptions.defaultOptions()));
        assertEquals("hello", tokenizer.decode(new long[] {1, 4, 2}, DecodeOptions.defaultOptions()));
        assertEquals("[CLS] hello [SEP]",
                tokenizer.decode(new long[] {1, 4, 2}, new DecodeOptions().skipSpecialTokens(false)));
    }

    @Test
    void loadsHuggingFaceWordPieceTokenizerJson() throws Exception {
        String json = """
                {
                  "normalizer": {"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,
                    "strip_accents":null,"lowercase":true},
                  "pre_tokenizer": {"type":"BertPreTokenizer"},
                  "model": {"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##",
                    "max_input_chars_per_word":100,
                    "vocab":{"[UNK]":0,"[CLS]":1,"[SEP]":2,"[PAD]":3,"hello":4,"world":5}},
                  "added_tokens":[
                    {"id":0,"content":"[UNK]","special":true},
                    {"id":1,"content":"[CLS]","special":true},
                    {"id":2,"content":"[SEP]","special":true},
                    {"id":3,"content":"[PAD]","special":true}
                  ]
                }
                """;
        Path directory = Files.createTempDirectory("bert-tokenizer");
        Path path = directory.resolve("tokenizer.json");
        try {
            Files.writeString(path, json);
            HuggingFaceWordPieceLoader loader = new HuggingFaceWordPieceLoader();
            org.junit.jupiter.api.Assertions.assertTrue(loader.supports(path));
            Tokenizer loaded = loader.load(path);
            assertArrayEquals(new long[] {1, 4, 5, 2},
                    loaded.encode("Hello world", new EncodeOptions().addBos(true).addEos(true)));
            assertEquals(6, loaded.vocabSize());
            assertEquals(3, loaded.padTokenId());
            assertEquals(true, TokenizerFactory.load(path, null) instanceof WordPieceTokenizer);
        } finally {
            Files.deleteIfExists(path);
            Files.deleteIfExists(directory);
        }
    }
}
