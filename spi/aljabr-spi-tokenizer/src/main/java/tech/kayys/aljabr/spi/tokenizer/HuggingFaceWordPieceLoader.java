package tech.kayys.aljabr.spi.tokenizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Loads Hugging Face tokenizer.json files whose model is WordPiece. */
public final class HuggingFaceWordPieceLoader {
    private final ObjectMapper mapper = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    public boolean supports(Path path) throws IOException {
        TokenizerJson json = mapper.readValue(path.toFile(), TokenizerJson.class);
        return json.model != null && "WordPiece".equalsIgnoreCase(json.model.type);
    }

    public Tokenizer load(Path path) {
        try {
            TokenizerJson json = mapper.readValue(path.toFile(), TokenizerJson.class);
            if (json.model == null || !"WordPiece".equalsIgnoreCase(json.model.type)
                    || json.model.vocab == null || json.model.vocab.isEmpty()) {
                throw new IOException("tokenizer.json does not contain a WordPiece vocabulary: " + path);
            }
            Map<String, Integer> vocabulary = new HashMap<>(json.model.vocab);
            Map<String, Integer> specialTokens = new HashMap<>();
            if (json.added_tokens != null) {
                for (TokenizerJson.AddedToken token : json.added_tokens) {
                    if (token.content == null) continue;
                    vocabulary.put(token.content, token.id);
                    if (token.special) specialTokens.put(token.content, token.id);
                }
            }
            String unknown = json.model.unk_token == null ? "[UNK]" : json.model.unk_token;
            Integer unknownId = vocabulary.get(unknown);
            if (unknownId == null) throw new IOException("WordPiece vocabulary is missing unk_token " + unknown);
            String prefix = json.model.continuing_subword_prefix == null
                    ? "##" : json.model.continuing_subword_prefix;
            boolean lowercase = json.normalizer == null || json.normalizer.lowercase == null
                    || json.normalizer.lowercase;
            boolean stripAccents = json.normalizer != null && Boolean.TRUE.equals(json.normalizer.strip_accents);
            boolean cleanText = json.normalizer == null || json.normalizer.clean_text == null
                    || json.normalizer.clean_text;
            int maxChars = json.model.max_input_chars_per_word > 0
                    ? json.model.max_input_chars_per_word : 100;
            return new WordPieceTokenizer(vocabulary, specialTokens, unknownId,
                    vocabulary.getOrDefault("[CLS]", -1), vocabulary.getOrDefault("[SEP]", -1),
                    vocabulary.getOrDefault("[PAD]", -1), prefix, lowercase, stripAccents,
                    cleanText, maxChars);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load Hugging Face WordPiece tokenizer: " + path, e);
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class TokenizerJson {
        public Model model;
        public Normalizer normalizer;
        public List<AddedToken> added_tokens;

        @JsonIgnoreProperties(ignoreUnknown = true)
        static final class Model {
            public String type;
            public Map<String, Integer> vocab;
            public String unk_token;
            public String continuing_subword_prefix;
            public int max_input_chars_per_word;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        static final class Normalizer {
            public Boolean lowercase;
            public Boolean strip_accents;
            public Boolean clean_text;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        static final class AddedToken {
            public int id;
            public String content;
            public boolean special;
        }
    }
}
