package tech.kayys.aljabr.spi.tokenizer;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * BERT-compatible basic-tokenizer plus greedy longest-match-first WordPiece.
 * The vocabulary and special-token IDs are supplied by a Hugging Face loader;
 * this implementation has no model-family or storage-format dependency.
 */
public final class WordPieceTokenizer implements Tokenizer {
    private final Map<String, Integer> vocabulary;
    private final Map<Integer, String> reverseVocabulary;
    private final Map<String, Integer> specialTokens;
    private final int unknownId, clsId, sepId, padId, maxCharsPerWord;
    private final String continuationPrefix;
    private final boolean lowercase, stripAccents, cleanText;

    public WordPieceTokenizer(Map<String, Integer> vocabulary, Map<String, Integer> specialTokens,
            int unknownId, int clsId, int sepId, int padId, String continuationPrefix,
            boolean lowercase, boolean stripAccents, boolean cleanText, int maxCharsPerWord) {
        this.vocabulary = Map.copyOf(Objects.requireNonNull(vocabulary, "vocabulary"));
        this.specialTokens = Map.copyOf(specialTokens == null ? Map.of() : specialTokens);
        if (unknownId < 0 || !this.vocabulary.containsValue(unknownId)) {
            throw new IllegalArgumentException("unknownId must identify a vocabulary entry");
        }
        if (maxCharsPerWord <= 0) throw new IllegalArgumentException("maxCharsPerWord must be positive");
        this.unknownId = unknownId;
        this.clsId = clsId;
        this.sepId = sepId;
        this.padId = padId;
        this.continuationPrefix = Objects.requireNonNull(continuationPrefix, "continuationPrefix");
        this.lowercase = lowercase;
        this.stripAccents = stripAccents;
        this.cleanText = cleanText;
        this.maxCharsPerWord = maxCharsPerWord;
        Map<Integer, String> reverse = new HashMap<>();
        this.vocabulary.forEach((token, id) -> reverse.put(id, token));
        this.reverseVocabulary = Map.copyOf(reverse);
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        if (text == null || text.isEmpty()) return new long[0];
        EncodeOptions effective = options == null ? EncodeOptions.defaultOptions() : options;
        List<Long> ids = new ArrayList<>();
        if (effective.addBos && clsId >= 0) ids.add((long) clsId);
        List<String> markers = new ArrayList<>(specialTokens.keySet());
        markers.sort((left, right) -> Integer.compare(right.length(), left.length()));
        int cursor = 0;
        while (cursor < text.length()) {
            String marker = null;
            for (String candidate : markers) {
                if (text.startsWith(candidate, cursor)) {
                    marker = candidate;
                    break;
                }
            }
            if (marker != null) {
                ids.add(specialTokens.get(marker).longValue());
                cursor += marker.length();
                continue;
            }
            int nextMarker = text.length();
            for (String candidate : markers) {
                int found = text.indexOf(candidate, cursor);
                if (found >= 0 && found < nextMarker) nextMarker = found;
            }
            String ordinary = text.substring(cursor, nextMarker);
            for (String token : basicTokens(ordinary)) ids.addAll(wordPiece(token));
            cursor = nextMarker;
        }
        if (effective.addEos && sepId >= 0) ids.add((long) sepId);
        long[] result = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) result[i] = ids.get(i);
        return result;
    }

    private List<Long> wordPiece(String token) {
        List<Long> result = new ArrayList<>();
        int[] codePoints = token.codePoints().toArray();
        if (codePoints.length > maxCharsPerWord) return List.of((long) unknownId);
        int start = 0;
        while (start < codePoints.length) {
            int end = codePoints.length;
            String matched = null;
            Integer matchedId = null;
            while (start < end) {
                String piece = new String(codePoints, start, end - start);
                if (start > 0) piece = continuationPrefix + piece;
                Integer id = vocabulary.get(piece);
                if (id != null) {
                    matched = piece;
                    matchedId = id;
                    break;
                }
                end--;
            }
            if (matched == null) return List.of((long) unknownId);
            result.add((long) matchedId);
            start = end;
        }
        return result;
    }

    private List<String> basicTokens(String input) {
        String normalized = normalize(input);
        List<String> result = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        normalized.codePoints().forEach(cp -> {
            if (isWhitespace(cp)) {
                flush(current, result);
            } else if (isPunctuation(cp)) {
                flush(current, result);
                result.add(new String(Character.toChars(cp)));
            } else {
                current.appendCodePoint(cp);
            }
        });
        flush(current, result);
        return result;
    }

    private String normalize(String input) {
        StringBuilder cleaned = new StringBuilder();
        input.codePoints().forEach(cp -> {
            if (cleanText && (cp == 0 || cp == 0xfffd || isControl(cp))) return;
            cleaned.appendCodePoint(isWhitespace(cp) ? 0x20 : cp);
        });
        String value = cleaned.toString();
        if (lowercase) value = value.toLowerCase(java.util.Locale.ROOT);
        if (stripAccents || lowercase) {
            value = Normalizer.normalize(value, Normalizer.Form.NFD)
                    .replaceAll("\\p{Mn}+", "");
        }
        return value;
    }

    private static void flush(StringBuilder current, List<String> output) {
        if (current.length() > 0) {
            output.add(current.toString());
            current.setLength(0);
        }
    }

    private static boolean isWhitespace(int cp) {
        return cp == 0x20 || cp == 0x09 || cp == 0x0a || cp == 0x0d
                || Character.getType(cp) == Character.SPACE_SEPARATOR;
    }

    private static boolean isControl(int cp) {
        if (cp == 0x09 || cp == 0x0a || cp == 0x0d) return false;
        int type = Character.getType(cp);
        return type == Character.CONTROL || type == Character.FORMAT;
    }

    private static boolean isPunctuation(int cp) {
        int type = Character.getType(cp);
        return (type >= Character.CONNECTOR_PUNCTUATION && type <= Character.OTHER_PUNCTUATION)
                || (type >= Character.INITIAL_QUOTE_PUNCTUATION
                    && type <= Character.FINAL_QUOTE_PUNCTUATION);
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        if (tokens == null || tokens.length == 0) return "";
        StringBuilder decoded = new StringBuilder();
        for (long rawId : tokens) {
            String token = reverseVocabulary.get((int) rawId);
            if (token == null || token.equals("[PAD]") || token.equals("[CLS]")
                    || token.equals("[SEP]") || token.equals("[MASK]")) continue;
            if (token.startsWith(continuationPrefix)) {
                decoded.append(token.substring(continuationPrefix.length()));
            } else {
                if (!decoded.isEmpty() && !isPunctuationToken(token)) decoded.append(' ');
                decoded.append(token);
            }
        }
        return decoded.toString();
    }

    private static boolean isPunctuationToken(String token) {
        if (token.codePointCount(0, token.length()) != 1) return false;
        return isPunctuation(token.codePointAt(0));
    }

    @Override public int vocabSize() { return vocabulary.size(); }
    @Override public int bosTokenId() { return clsId; }
    @Override public int eosTokenId() { return sepId; }
    @Override public int padTokenId() { return padId; }
    @Override public int[] allStopTokenIds() { return sepId >= 0 ? new int[] {sepId} : new int[0]; }
    @Override public Map<String, Integer> specialTokens() { return specialTokens; }
}
