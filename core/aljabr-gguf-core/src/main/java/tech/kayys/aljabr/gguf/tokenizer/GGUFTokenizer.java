package tech.kayys.aljabr.gguf.tokenizer;

import org.jboss.logging.Logger;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.spi.tokenizer.BpeTokenizer;
import tech.kayys.aljabr.spi.tokenizer.Gpt2PreTokenizer;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Adapter that initializes a BpeTokenizer from GGUF metadata.
 *
 * This implementation is defensive and tries to support a variety of GGUF
 * converter outputs and future model/tokenizer styles (GPT-2 byte-level,
 * raw UTF-8 tokenizers like Qwen/Tiktoken, and any converters that emit
 * alternate metadata keys).
 * @author bhangun
 */
public final class GGUFTokenizer implements Tokenizer {
    private static final Logger log = Logger.getLogger(GGUFTokenizer.class);

    private final Tokenizer delegate;
    private final int vocabSize;
    private final int bosId;
    private final int eosId;
    private final int padId;
    private final boolean addBos;
    private final boolean addEos;

    private final Map<String, Integer> specialTokens;
    private final List<Integer> additionalEosIds;

    public GGUFTokenizer(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        Map<String, Object> meta = model.metadata();

        // Token list (several GGUF variants may use slightly different keys)
        @SuppressWarnings("unchecked")
        List<String> tokens = (List<String>) meta.getOrDefault("tokenizer.ggml.tokens",
                meta.get("tokenizer.tokens"));
        if (tokens == null || tokens.isEmpty()) {
            throw new IllegalArgumentException("GGUF model is missing mandatory tokenizer tokens metadata");
        }

        @SuppressWarnings("unchecked")
        List<String> merges = (List<String>) meta.get("tokenizer.ggml.merges");
        if (merges == null) {
            // Some exporters put merges under other keys
            merges = (List<String>) meta.get("tokenizer.merges");
        }

        // token_type can be numbers or strings depending on converter
        @SuppressWarnings("unchecked")
        List<Object> tokenTypes = (List<Object>) meta.get("tokenizer.ggml.token_type");
        if (tokenTypes == null) {
            tokenTypes = (List<Object>) meta.get("tokenizer.token_type");
        }

        Map<String, Integer> tokenToId = new HashMap<>();
        Map<Integer, String> idToToken = new HashMap<>();
        for (int i = 0; i < tokens.size(); i++) {
            String tok = tokens.get(i);
            tokenToId.put(tok, i);
            idToToken.put(i, tok);
        }

        Map<String, Integer> mergeRanks = new HashMap<>();
        if (merges != null) {
            for (int i = 0; i < merges.size(); i++) {
                // Keep the space delimiter for BpeTokenizer compatibility
                String key = merges.get(i).trim();
                mergeRanks.put(key, i);
            }
        }

        this.vocabSize = tokens.size();

        // Support multiple metadata key names and fallbacks
        this.bosId = firstMetaInt(meta, new String[]{"tokenizer.ggml.bos_token_id", "tokenizer.bos_token_id", "bos_token_id"}, 1);
        this.eosId = firstMetaInt(meta, new String[]{"tokenizer.ggml.eos_token_id", "tokenizer.eos_token_id", "eos_token_id"}, 2);
        this.padId = firstMetaInt(meta, new String[]{"tokenizer.ggml.padding_token_id", "tokenizer.padding_token_id", "pad_token_id"}, -1);

        this.addBos = firstMetaBool(meta, new String[]{"tokenizer.ggml.add_bos_token", "tokenizer.add_bos_token"}, false);
        this.addEos = firstMetaBool(meta, new String[]{"tokenizer.ggml.add_eos_token", "tokenizer.add_eos_token"}, false);

        String arch = String.valueOf(meta.getOrDefault("general.architecture", meta.getOrDefault("architecture", "llama")));
        String tokenizerModel = String.valueOf(meta.getOrDefault("tokenizer.ggml.model", meta.getOrDefault("tokenizer.model", "gpt2")));

        boolean isRawUtf8Vocab = detectRawUtf8Vocab(meta, arch, tokenizerModel, tokens);

        log.debugf("GGUF Tokenizer init: model=%s arch=%s rawUtf8=%b vocab=%d", tokenizerModel, arch, isRawUtf8Vocab, tokens.size());

        // Build special token map from explicit metadata if present
        this.specialTokens = new HashMap<>();
        this.additionalEosIds = new ArrayList<>();

        // Preferred: explicit special token mapping metadata
        @SuppressWarnings("unchecked")
        Map<String, Object> specials = (Map<String, Object>) meta.get("tokenizer.special_tokens");
        if (specials == null) specials = (Map<String, Object>) meta.get("special_tokens");

        if (specials != null) {
            for (Map.Entry<String, Object> e : specials.entrySet()) {
                Object v = e.getValue();
                Integer id = null;
                if (v instanceof Number) id = ((Number) v).intValue();
                else if (v instanceof String) {
                    try { id = Integer.parseInt((String) v); } catch (NumberFormatException ex) { /* ignore */ }
                }
                if (id != null) {
                    specialTokens.put(e.getKey(), id);
                    if (isLikelyEosName(e.getKey())) additionalEosIds.add(id);
                }
            }
        }

        // Fallback: infer special tokens from token_type array
        if (tokenTypes != null) {
            for (int i = 0; i < Math.min(tokens.size(), tokenTypes.size()); i++) {
                Object tt = tokenTypes.get(i);
                if (isControlToken(tt, tokens.get(i))) {
                    specialTokens.put(tokens.get(i), i);
                    if (tokens.get(i).toLowerCase().contains("im_end") || tokens.get(i).toLowerCase().contains("endoftext") || isLikelyEosName(tokens.get(i))) {
                        additionalEosIds.add(i);
                    }
                }
            }
            log.debugf("Inferred %d special tokens from token_type metadata (additional EOS: %s)", specialTokens.size(), additionalEosIds);
        }

        // Final fallback: Ensure known Gemma control tokens are treated as special if they exist in vocab
        if (arch.toLowerCase().startsWith("gemma")) {
            String[] gemmaSpecials = {
                    "<start_of_turn>", "<end_of_turn>", "<|turn>", "<turn|>", "<bos>", "<eos>", "<pad>"
            };
            for (String s : gemmaSpecials) {
                if (!specialTokens.containsKey(s) && tokenToId.containsKey(s)) {
                    int id = tokenToId.get(s);
                    specialTokens.put(s, id);
                    if (s.contains("eos") || s.contains("end_of_turn")) additionalEosIds.add(id);
                    log.debugf("Forced Gemma special token: %s -> %d", s, id);
                }
            }
        }

        // Build byte encoder (used by BPE implementation) - kept backward-compatible
        Map<Character, String> byteEncoder = createByteEncoder();

        // Determine unk token id if present
        int unkId = firstMetaInt(meta, new String[]{"tokenizer.ggml.unk_token_id", "tokenizer.unk_token_id", "unk_token_id"}, -1);

        this.delegate = new BpeTokenizer(
                tokenToId,
                idToToken,
                mergeRanks,
                isRawUtf8Vocab,
                unkId,
                bosId,
                eosId,
                padId,
                specialTokens
        );
        log.debugf("GGUFTokenizer initialized with %d special tokens: %s", specialTokens.size(), specialTokens);
    }

    private static boolean isLikelyEosName(String name) {
        String n = name.toLowerCase();
        return n.contains("end") || n.contains("eos") || n.contains("im_end") || n.contains("</s>") || n.contains("endoftext");
    }

    private static boolean isControlToken(Object tokenType, String tokenStr) {
        if (tokenType instanceof Number) {
            int v = ((Number) tokenType).intValue();
            // many converters use 3 for control/special tokens
            return v == 3 || v == 4 || v == 5;
        } else if (tokenType instanceof String) {
            String s = ((String) tokenType).toLowerCase();
            return s.contains("control") || s.contains("special") || s.contains("user_defined");
        }
        // Heuristic: tokens that look like control tokens (start/end markers)
        String t = tokenStr.toLowerCase();
        return t.startsWith("<|") || t.startsWith("</") || t.contains("endof") || t.contains("im_end");
    }

    /**
     * Heuristic to detect whether tokens are raw UTF-8 strings or GPT-2 byte-level encodings.
     */
    private boolean detectRawUtf8Vocab(Map<String, Object> meta, String arch, String tokenizerModel, List<String> tokens) {
        // 1) explicit override metadata (strongest signal)
        Object fmt = meta.get("tokenizer.format");
        if (fmt == null) fmt = meta.get("tokenizer.encoding");
        if (fmt instanceof String) {
            String f = ((String) fmt).toLowerCase();
            if (f.contains("tiktoken") || f.contains("qwen") || f.contains("utf-8") || f.contains("utf8")) return true;
            if (f.contains("gpt2") || f.contains("byte") || f.contains("byte-level")) return false;
        }

        // 2) name hints (lightweight)
        String nameHint = (arch + " " + tokenizerModel).toLowerCase();
        double nameScore = 0.0;
        // negative for signals that imply byte-level, positive for raw UTF-8
        String[] rawHints = new String[]{"qwen", "tiktoken", "utf8", "cl100k", "cl100k_base"};
        String[] byteHints = new String[]{"gpt2", "byte", "byte-level", "gpt-2"};
        for (String h : rawHints) if (nameHint.contains(h)) nameScore += 1.0;
        for (String h : byteHints) if (nameHint.contains(h)) nameScore -= 2.0;

        // 3) sampled token-content statistics (single pass up to N tokens)
        int N = Math.min(2000, tokens.size());
        int spaces = 0, gptMarkers = 0, sentencePieceMarkers = 0; long totalLen = 0;
        for (int i = 0; i < N; i++) {
            String t = tokens.get(i);
            totalLen += t.length();
            if (t.indexOf('\u0120') >= 0 || t.indexOf('\u010A') >= 0) gptMarkers++;
            if (t.indexOf('\u2581') >= 0) sentencePieceMarkers++;
            if (t.indexOf(' ') >= 0) spaces++;
        }
        double pctSpaces = (double) spaces / Math.max(1, N);
        double pctGpt = (double) gptMarkers / Math.max(1, N);
        double pctSentencePiece = (double) sentencePieceMarkers / Math.max(1, N);
        double avgLen = (double) totalLen / Math.max(1, N);

        // 4) weighted scoring
        // Weights chosen to prioritize GPT markers as a strong byte-level signal
        double score = 0.0;
        score += pctSpaces * 3.0;    // spaces -> raw UTF-8
        score += pctSentencePiece * 6.0; // SentencePiece "word boundary" marker -> raw UTF-8
        score -= pctGpt * 6.0;       // GPT markers -> byte-level (strong)
        score += Math.min(1.0, Math.max(0.0, (avgLen - 2.0) / 10.0)) * 0.5; // longer tokens hint raw
        score += nameScore * 0.5;    // name hints are weaker

        // 5) meta overrides that suggest byte-level explicitly
        Object modelEnc = meta.get("tokenizer.ggml.model");
        if (modelEnc instanceof String) {
            String me = ((String) modelEnc).toLowerCase();
            if (me.contains("tiktoken") || me.contains("qwen")) score += 5.0; // Tiktoken/Qwen strongly implies raw
            if ((me.contains("gpt2") || me.contains("byte")) && pctSentencePiece < 0.01) score -= 4.0;
        }

        // 6) architecture-aware overrides
        String archLower = arch.toLowerCase();
        if (archLower.contains("qwen")) score += 10.0;
        if (archLower.startsWith("gemma") && pctSentencePiece > 0.01) score += 8.0;

        log.debugf(
                "Tokenizer detection: pctSpaces=%.3f pctSentencePiece=%.3f pctGpt=%.3f avgLen=%.2f nameScore=%.2f score=%.3f",
                pctSpaces, pctSentencePiece, pctGpt, avgLen, nameScore, score);

        // Tunable threshold; conservative default: require score > 0.5 to treat as raw UTF-8
        return score > 0.5;
    }

    public boolean shouldAddBos() { return addBos; }
    public boolean shouldAddEos() { return addEos; }

    public boolean isEosToken(int tokenId) {
        if (tokenId == eosId) return true;
        return additionalEosIds.contains(tokenId);
    }

    public List<Integer> getAdditionalEosIds() { return List.copyOf(additionalEosIds); }

    private int firstMetaInt(Map<String, Object> meta, String[] keys, int def) {
        for (String k : keys) {
            Object v = meta.get(k);
            if (v instanceof Number) return ((Number) v).intValue();
            if (v instanceof String) {
                try { return Integer.parseInt((String) v); } catch (NumberFormatException ignored) {}
            }
        }
        return def;
    }

    private boolean firstMetaBool(Map<String, Object> meta, String[] keys, boolean def) {
        for (String k : keys) {
            Object v = meta.get(k);
            if (v instanceof Boolean) return (Boolean) v;
            if (v instanceof Number) return ((Number) v).intValue() != 0;
            if (v instanceof String) {
                String s = ((String) v).toLowerCase();
                if (s.equals("true") || s.equals("1")) return true;
                if (s.equals("false") || s.equals("0")) return false;
            }
        }
        return def;
    }

    private Map<Character, String> createByteEncoder() {
        Map<Character, String> encoder = new HashMap<>();
        List<Integer> bs = new ArrayList<>();
        // Range 33..126
        for (int i = 33; i <= 126; i++) bs.add(i);
        // Range 161..172
        for (int i = 161; i <= 172; i++) bs.add(i);
        // Range 174..255
        for (int i = 174; i <= 255; i++) bs.add(i);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < 256; i++) {
            encoder.put((char) bs.get(i).intValue(), String.valueOf((char) cs.get(i).intValue()));
        }
        return encoder;
    }

    @Override public long[] encode(String text, EncodeOptions options) { return delegate.encode(text, options); }
    @Override public String decode(long[] tokens, DecodeOptions options) { return delegate.decode(tokens, options); }
    @Override public String decode(long[] tokens, int offset, int length, DecodeOptions options) {
        return delegate.decode(tokens, offset, length, options);
    }
    @Override public int vocabSize() { return vocabSize; }
    @Override public int bosTokenId() { return bosId; }
    @Override public int eosTokenId() { return eosId; }
    @Override public int padTokenId() { return padId; }

    @Override
    public int[] allStopTokenIds() {
        List<Integer> all = new ArrayList<>();
        if (eosId >= 0) all.add(eosId);
        all.addAll(additionalEosIds);
        return all.stream().mapToInt(i -> i).distinct().toArray();
    }

    @Override
    public java.util.Map<String, Integer> specialTokens() {
        return delegate.specialTokens();
    }
}
