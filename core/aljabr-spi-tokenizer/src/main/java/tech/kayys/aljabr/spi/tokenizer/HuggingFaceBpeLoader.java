/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HuggingFaceBpeLoader {

    public Tokenizer load(Path path) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
            TokenizerJson json = mapper.readValue(path.toFile(), TokenizerJson.class);

            Map<String, Integer> tokenToId = json.model != null && json.model.vocab != null
                    ? new HashMap<>(json.model.vocab)
                    : new HashMap<>();

            Map<String, Integer> specialTokens = new HashMap<>();
            if (json.added_tokens != null) {
                for (TokenizerJson.AddedToken at : json.added_tokens) {
                    if (at.content != null) {
                        tokenToId.put(at.content, at.id);
                        if (at.special) {
                            specialTokens.put(at.content, at.id);
                        }
                    }
                }
            }

            Map<Integer, String> idToToken = new HashMap<>();
            tokenToId.forEach((k, v) -> idToToken.put(v, k));

            Map<String, Integer> mergeRanks = new HashMap<>();
            if (json.model != null && json.model.merges != null) {
                for (int i = 0; i < json.model.merges.size(); i++) {
                    Object mergeObj = json.model.merges.get(i);
                    String part0 = null;
                    String part1 = null;
                    if (mergeObj instanceof String mergeStr) {
                        mergeRanks.put(mergeStr, i);
                        String[] parts = mergeStr.split(" ", 2);
                        if (parts.length == 2) {
                            part0 = parts[0];
                            part1 = parts[1];
                        }
                    } else if (mergeObj instanceof List<?> list && list.size() >= 2) {
                        part0 = String.valueOf(list.get(0));
                        part1 = String.valueOf(list.get(1));
                    }
                    if (part0 != null && part1 != null) {
                        mergeRanks.put(part0 + " " + part1, i);
                        mergeRanks.put(part0 + part1, i);
                    }
                }
            }

            int bosId = findToken(tokenToId, "<bos>", "<s>", "<|im_start|>", "<|begin_of_text|>", "<|endoftext|>");
            int eosId = findToken(tokenToId, "<eos>", "</s>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>");
            int padId = findToken(tokenToId, "<pad>", "<|pad|>", "<|endoftext|>");
            int unkId = findToken(tokenToId, "<unk>", "<|unk|>");

            return new BpeTokenizer(
                    tokenToId,
                    idToToken,
                    mergeRanks,
                    false, // rawUtf8Vocab
                    unkId,
                    bosId,
                    eosId,
                    padId,
                    specialTokens);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load tokenizer: " + path, e);
        }
    }

    private int findToken(Map<String, Integer> tokenToId, String... candidates) {
        for (String c : candidates) {
            Integer id = tokenToId.get(c);
            if (id != null) {
                return id;
            }
        }
        return -1;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TokenizerJson {
        public Model model;
        public List<AddedToken> added_tokens;

        @JsonIgnoreProperties(ignoreUnknown = true)
        public static class Model {
            public String type;
            public Map<String, Integer> vocab;
            public List<Object> merges;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        public static class AddedToken {
            public int id;
            public String content;
            public boolean special;
        }
    }
}
