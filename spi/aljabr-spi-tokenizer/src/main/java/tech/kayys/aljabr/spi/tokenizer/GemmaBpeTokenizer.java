/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * BPE tokenizer for the serialized Gemma tokenizer pipeline.
 *
 * <p>This implementation uses the Gemma whitespace marker ({@code ▁}) and
 * Hugging Face byte-fallback tokens ({@code <0xHH>}). It deliberately does
 * not use GPT-2 byte remapping or GPT-2 pre-tokenization.</p>
 */
public final class GemmaBpeTokenizer implements Tokenizer {
    private static final Pattern BYTE_TOKEN = Pattern.compile("<0x([0-9A-Fa-f]{2})>");
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> inverse;
    private final Map<String, Integer> merges;
    private final Map<String, Integer> specials;
    private final int unkId;
    private final int bosId;
    private final int eosId;
    private final int padId;

    GemmaBpeTokenizer(Map<String, Integer> vocab, List<String> mergeList,
                      Map<String, Integer> specials, int unkId, int bosId, int eosId, int padId) {
        this.vocab = Map.copyOf(vocab);
        this.inverse = new HashMap<>();
        vocab.forEach((token, id) -> inverse.put(id, token));
        this.merges = new HashMap<>();
        for (int i = 0; i < mergeList.size(); i++) {
            this.merges.put(mergeList.get(i), i);
        }
        this.specials = Map.copyOf(specials);
        this.unkId = unkId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.padId = padId;
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        if (text == null) return new long[0];
        EncodeOptions opts = options == null ? EncodeOptions.defaultOptions() : options;
        List<Long> ids = new ArrayList<>();
        if (opts.addBos && bosId >= 0) ids.add((long) bosId);

        List<String> specialStrings = new ArrayList<>(specials.keySet());
        specialStrings.sort(Comparator.comparingInt(String::length).reversed());
        Pattern specialPattern = specialStrings.isEmpty() ? null
                : Pattern.compile(specialStrings.stream().map(Pattern::quote)
                        .collect(java.util.stream.Collectors.joining("|")));
        Matcher matcher = specialPattern == null ? null : specialPattern.matcher(text);
        int cursor = 0;
        while (matcher != null && matcher.find()) {
            encodeOrdinary(text.substring(cursor, matcher.start()), ids);
            ids.add(specials.get(matcher.group()).longValue());
            cursor = matcher.end();
        }
        encodeOrdinary(text.substring(cursor), ids);
        if (opts.addEos && eosId >= 0) ids.add((long) eosId);
        return ids.stream().mapToLong(Long::longValue).toArray();
    }

    private void encodeOrdinary(String input, List<Long> output) {
        if (input.isEmpty()) return;
        String normalized = input.replace(" ", "▁");
        List<String> symbols = new ArrayList<>();
        for (int offset = 0; offset < normalized.length();) {
            int cp = normalized.codePointAt(offset);
            String symbol = new String(Character.toChars(cp));
            if (vocab.containsKey(symbol)) {
                symbols.add(symbol);
            } else {
                byte[] bytes = symbol.getBytes(StandardCharsets.UTF_8);
                for (byte value : bytes) {
                    String byteToken = String.format("<0x%02X>", value & 0xff);
                    if (vocab.containsKey(byteToken)) symbols.add(byteToken);
                    else symbols.add(symbol);
                }
            }
            offset += Character.charCount(cp);
        }
        while (symbols.size() > 1) {
            int best = Integer.MAX_VALUE;
            int bestIndex = -1;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = merges.get(symbols.get(i) + " " + symbols.get(i + 1));
                if (rank != null && rank < best) {
                    best = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;
            symbols.set(bestIndex, symbols.get(bestIndex) + symbols.get(bestIndex + 1));
            symbols.remove(bestIndex + 1);
        }
        for (String symbol : symbols) {
            Integer id = vocab.get(symbol);
            if (id != null) output.add(id.longValue());
            else if (unkId >= 0) output.add((long) unkId);
            else throw new IllegalArgumentException("Gemma tokenizer has no token or byte fallback for: " + symbol);
        }
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        if (tokens == null || tokens.length == 0) return "";
        DecodeOptions opts = options == null ? DecodeOptions.defaultOptions() : options;
        StringBuilder result = new StringBuilder();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (long tokenId : tokens) {
            String token = inverse.get((int) tokenId);
            if (token == null) continue;
            if (opts.skipSpecialTokens && specials.containsKey(token)) continue;
            Matcher byteMatcher = BYTE_TOKEN.matcher(token);
            if (byteMatcher.matches()) {
                bytes.write(Integer.parseInt(byteMatcher.group(1), 16));
                continue;
            }
            flushBytes(bytes, result);
            result.append(token);
        }
        flushBytes(bytes, result);
        return result.toString().replace('▁', ' ');
    }

    private static void flushBytes(ByteArrayOutputStream bytes, StringBuilder target) {
        if (bytes.size() == 0) return;
        target.append(bytes.toString(StandardCharsets.UTF_8));
        bytes.reset();
    }

    @Override public int vocabSize() { return vocab.size(); }
    @Override public int bosTokenId() { return bosId; }
    @Override public int eosTokenId() { return eosId; }
    @Override public int padTokenId() { return padId; }
    @Override public int[] allStopTokenIds() { return eosId < 0 ? new int[0] : new int[]{eosId}; }
    @Override public Map<String, Integer> specialTokens() { return specials; }
}
