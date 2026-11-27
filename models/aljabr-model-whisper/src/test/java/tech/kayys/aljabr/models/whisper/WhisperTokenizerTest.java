package tech.kayys.aljabr.models.whisper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class WhisperTokenizerTest {

    @TempDir
    Path modelDirectory;

    @Test
    void buildsTranscriptionAndTranslationPromptsFromHuggingFaceTokenizer() throws Exception {
        writeTokenizer();
        WhisperTokenizer tokenizer = WhisperTokenizer.load(modelDirectory);

        assertArrayEquals(new long[]{10, 11, 12, 13},
                tokenizer.decoderPrompt("en", WhisperTokenizer.Task.TRANSCRIBE, false));
        assertArrayEquals(new long[]{10, 14, 15},
                tokenizer.decoderPrompt("id", WhisperTokenizer.Task.TRANSLATE, true));
    }

    @Test
    void removesWhisperControlTokensFromDecodedTranscript() throws Exception {
        writeTokenizer();
        WhisperTokenizer tokenizer = WhisperTokenizer.load(modelDirectory);

        String decoded = tokenizer.decodeGeneratedText(new long[]{1, 2, 3, 10, 11, 12, 16});

        assertEquals("hi!", decoded);
    }

    @Test
    void rejectsUnknownLanguageAndMissingWhisperTokens() throws Exception {
        writeTokenizer();
        WhisperTokenizer tokenizer = WhisperTokenizer.load(modelDirectory);

        assertThrows(IllegalArgumentException.class,
                () -> tokenizer.decoderPrompt("xx", WhisperTokenizer.Task.TRANSCRIBE, false));
        assertThrows(IllegalArgumentException.class,
                () -> tokenizer.decoderPrompt("not a language", WhisperTokenizer.Task.TRANSCRIBE, false));
    }

    private void writeTokenizer() throws Exception {
        Files.writeString(modelDirectory.resolve("tokenizer.json"), """
                {
                  "version": "1.0",
                  "model": {
                    "type": "BPE",
                    "vocab": {
                      "h": 1, "i": 2, "!": 3,
                      "<|startoftranscript|>": 10,
                      "<|en|>": 11, "<|transcribe|>": 12,
                      "<|notimestamps|>": 13, "<|id|>": 14,
                      "<|translate|>": 15, "<|endoftext|>": 16
                    },
                    "merges": []
                  },
                  "added_tokens": [
                    {"id":10,"content":"<|startoftranscript|>","special":true},
                    {"id":11,"content":"<|en|>","special":true},
                    {"id":12,"content":"<|transcribe|>","special":true},
                    {"id":13,"content":"<|notimestamps|>","special":true},
                    {"id":14,"content":"<|id|>","special":true},
                    {"id":15,"content":"<|translate|>","special":true},
                    {"id":16,"content":"<|endoftext|>","special":true}
                  ]
                }
                """);
    }
}
