package tech.kayys.aljabr.models.whisper;

import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;
import tech.kayys.aljabr.spi.tokenizer.TokenizerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Whisper task-aware tokenizer facade.
 *
 * <p>Reuses Aljabr's Hugging Face tokenizer implementation and adds the
 * Whisper-specific decoder prefix and generated-text filtering. It does not
 * duplicate byte-level BPE or own vocabulary parsing.</p>
 */
public final class WhisperTokenizer {

    public enum Task {
        TRANSCRIBE("<|transcribe|>"),
        TRANSLATE("<|translate|>");

        private final String token;

        Task(String token) {
            this.token = token;
        }

        public String token() {
            return token;
        }
    }

    private static final String START_OF_TRANSCRIPT = "<|startoftranscript|>";
    private static final String END_OF_TEXT = "<|endoftext|>";
    private static final String NO_TIMESTAMPS = "<|notimestamps|>";

    private final Tokenizer delegate;
    private final Map<String, Integer> specialTokens;
    private final Set<Integer> nonTextTokenIds;

    public WhisperTokenizer(Tokenizer delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.specialTokens = Map.copyOf(delegate.specialTokens());
        this.nonTextTokenIds = new HashSet<>(specialTokens.values());
        requireToken(START_OF_TRANSCRIPT);
        requireToken(END_OF_TEXT);
        requireToken(Task.TRANSCRIBE.token());
        requireToken(Task.TRANSLATE.token());
        requireToken(NO_TIMESTAMPS);
    }

    public static WhisperTokenizer load(Path modelDirectory) throws IOException {
        return new WhisperTokenizer(TokenizerFactory.load(modelDirectory, null));
    }

    /** Build the decoder prefix required by multilingual Whisper checkpoints. */
    public long[] decoderPrompt(String language, Task task, boolean timestamps) {
        Objects.requireNonNull(task, "task");
        String normalizedLanguage = normalizeLanguage(language);
        int languageId = requireToken("<|" + normalizedLanguage + "|>");
        int startId = requireToken(START_OF_TRANSCRIPT);
        int taskId = requireToken(task.token());
        if (timestamps) {
            return new long[]{startId, languageId, taskId};
        }
        return new long[]{startId, languageId, taskId, requireToken(NO_TIMESTAMPS)};
    }

    public long[] encodeText(String text) {
        return delegate.encode(text, EncodeOptions.defaultOptions());
    }

    /**
     * Decode generated transcript tokens, excluding Whisper control, language,
     * timestamp, and task markers from user-visible text.
     */
    public String decodeGeneratedText(long[] generatedTokens) {
        Objects.requireNonNull(generatedTokens, "generatedTokens");
        long[] textTokens = new long[generatedTokens.length];
        int count = 0;
        for (long token : generatedTokens) {
            if (!nonTextTokenIds.contains((int) token)) {
                textTokens[count++] = token;
            }
            if (token == requireToken(END_OF_TEXT)) {
                break;
            }
        }
        return delegate.decode(java.util.Arrays.copyOf(textTokens, count),
                DecodeOptions.defaultOptions().skipSpecialTokens(true));
    }

    public int endOfTextTokenId() {
        return requireToken(END_OF_TEXT);
    }

    public Map<String, Integer> specialTokens() {
        return specialTokens;
    }

    private int requireToken(String token) {
        Integer id = specialTokens.get(token);
        if (id == null) {
            throw new IllegalArgumentException("Whisper tokenizer is missing required special token: " + token);
        }
        return id;
    }

    private static String normalizeLanguage(String language) {
        if (language == null || language.isBlank()) {
            throw new IllegalArgumentException("Whisper language code must not be blank");
        }
        String normalized = language.trim().toLowerCase(Locale.ROOT);
        if (!normalized.matches("[a-z]{2,3}")) {
            throw new IllegalArgumentException("Expected a Whisper language code such as 'en' or 'id': " + language);
        }
        return normalized;
    }
}
