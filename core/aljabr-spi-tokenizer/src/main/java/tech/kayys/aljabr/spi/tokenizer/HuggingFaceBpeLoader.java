/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HuggingFaceBpeLoader {

    public Tokenizer load(Path path) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            TokenizerJson json = mapper.readValue(path.toFile(), TokenizerJson.class);

            Map<String, Integer> tokenToId = json.model.vocab != null ? json.model.vocab : new HashMap<>();
            Map<Integer, String> idToToken = new HashMap<>();
            tokenToId.forEach((k, v) -> idToToken.put(v, k));

            Map<String, Integer> mergeRanks = new HashMap<>();
            if (json.model.merges != null) {
                for (int i = 0; i < json.model.merges.size(); i++) {
                    String[] parts = json.model.merges.get(i).split(" ");
                    if (parts.length >= 2) {
                        mergeRanks.put(parts[0] + parts[1], i);
                    }
                }
            }

            return new BpeTokenizer(
                    tokenToId,
                    idToToken,
                    mergeRanks,
                    buildByteEncoder(),
                    findToken(json, "<s>"),
                    findToken(json, "</s>"),
                    findToken(json, "<pad>"),
                    findToken(json, "<unk>"),
                    new Gpt2PreTokenizer());

        } catch (Exception e) {
            throw new RuntimeException("Failed to load tokenizer: " + path, e);
        }
    }

    private int findToken(TokenizerJson json, String token) {
        return json.model != null && json.model.vocab != null ? json.model.vocab.getOrDefault(token, -1) : -1;
    }

    private Map<Character, String> buildByteEncoder() {
        Map<Character, String> encoder = new HashMap<>();
        for (int i = 0; i < 256; i++) {
            encoder.put((char) i, String.valueOf((char) i));
        }
        return encoder;
    }

    public static class TokenizerJson {
        public Model model;

        public static class Model {
            public String type;
            public Map<String, Integer> vocab;
            public List<String> merges;
        }
    }
}
