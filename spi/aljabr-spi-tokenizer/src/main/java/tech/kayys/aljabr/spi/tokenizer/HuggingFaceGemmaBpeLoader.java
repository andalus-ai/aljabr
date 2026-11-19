/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads the Gemma-compatible Hugging Face BPE tokenizer.json pipeline.
 *
 * <p>The loader fails closed when the serialized model does not declare BPE
 * byte fallback. It does not silently reinterpret GPT-2 or SentencePiece
 * tokenizer files as Gemma tokenizers.</p>
 */
public final class HuggingFaceGemmaBpeLoader {
    private final ObjectMapper mapper = new ObjectMapper();

    public boolean supports(Path path) throws IOException {
        JsonNode root = mapper.readTree(path.toFile());
        JsonNode model = root.path("model");
        if (!"BPE".equals(model.path("type").asText())
                || !model.path("byte_fallback").asBoolean(false)
                || !model.path("fuse_unk").asBoolean(false)) return false;
        try {
            validatePipeline(root.path("normalizer"), root.path("pre_tokenizer"), root.path("decoder"));
            return true;
        } catch (IllegalArgumentException unsupported) {
            return false;
        }
    }

    public Tokenizer load(Path path) {
        try {
            JsonNode root = mapper.readTree(path.toFile());
            JsonNode model = root.path("model");
            if (!"BPE".equals(model.path("type").asText())) {
                throw new IllegalArgumentException("Expected tokenizer.json model.type=BPE");
            }
            if (!model.path("byte_fallback").asBoolean(false)) {
                throw new IllegalArgumentException("Expected BPE byte_fallback=true");
            }
            if (!model.path("fuse_unk").asBoolean(false)) {
                throw new IllegalArgumentException("Expected BPE fuse_unk=true");
            }
            validatePipeline(root.path("normalizer"), root.path("pre_tokenizer"), root.path("decoder"));

            Map<String, Integer> vocab = new HashMap<>();
            JsonNode vocabNode = model.path("vocab");
            vocabNode.fields().forEachRemaining(entry -> vocab.put(entry.getKey(), entry.getValue().asInt()));
            if (vocab.isEmpty()) throw new IllegalArgumentException("Tokenizer vocabulary is empty");

            Map<String, Integer> specials = new HashMap<>();
            JsonNode added = root.path("added_tokens");
            if (added.isArray()) {
                for (JsonNode item : added) {
                    if (item.path("special").asBoolean(false)) {
                        specials.put(item.path("content").asText(), item.path("id").asInt());
                    }
                }
            }
            for (String token : List.of("<bos>", "<eos>", "<pad>", "<unk>")) {
                if (vocab.containsKey(token)) specials.putIfAbsent(token, vocab.get(token));
            }
            List<String> merges = new ArrayList<>();
            JsonNode mergeNode = model.path("merges");
            if (mergeNode.isArray()) {
                for (JsonNode merge : mergeNode) {
                    if (merge.isTextual()) merges.add(merge.asText());
                    else if (merge.isArray() && merge.size() == 2) {
                        merges.add(merge.get(0).asText() + " " + merge.get(1).asText());
                    } else {
                        throw new IllegalArgumentException("Unsupported BPE merge entry: " + merge);
                    }
                }
            }
            return new GemmaBpeTokenizer(vocab, merges, specials,
                    vocab.getOrDefault("<unk>", -1), vocab.getOrDefault("<bos>", -1),
                    vocab.getOrDefault("<eos>", -1), vocab.getOrDefault("<pad>", -1));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load Gemma tokenizer: " + path, e);
        }
    }

    private static void validatePipeline(JsonNode normalizer, JsonNode preTokenizer, JsonNode decoder) {
        if (!containsReplace(normalizer, " ", "▁")) {
            throw new IllegalArgumentException("Unsupported Gemma normalizer: expected Replace(' ' -> '▁')");
        }
        if (!containsSplit(preTokenizer)) {
            throw new IllegalArgumentException("Unsupported Gemma pre-tokenizer: expected Split(' ', MergedWithPrevious)");
        }
        if (!containsDecoderStages(decoder)) {
            throw new IllegalArgumentException("Unsupported Gemma decoder: expected Replace('▁' -> ' '), ByteFallback, Fuse");
        }
    }

    private static boolean containsReplace(JsonNode node, String from, String to) {
        if (node == null || node.isMissingNode()) return false;
        if ("Replace".equals(node.path("type").asText())
                && from.equals(node.path("pattern").path("String").asText())
                && to.equals(node.path("content").asText())) return true;
        for (JsonNode child : node) if (containsReplace(child, from, to)) return true;
        return false;
    }

    private static boolean containsSplit(JsonNode node) {
        if (node == null || node.isMissingNode()) return false;
        if ("Split".equals(node.path("type").asText())
                && " ".equals(node.path("pattern").path("String").asText())
                && "MergedWithPrevious".equals(node.path("behavior").asText())) return true;
        for (JsonNode child : node) if (containsSplit(child)) return true;
        return false;
    }

    private static boolean containsDecoderStages(JsonNode node) {
        if (node == null || node.isMissingNode()) return false;
        boolean replace = containsReplace(node, "▁", " ");
        boolean fallback = containsType(node, "ByteFallback");
        boolean fuse = containsType(node, "Fuse");
        return replace && fallback && fuse;
    }

    private static boolean containsType(JsonNode node, String type) {
        if (node == null || node.isMissingNode()) return false;
        if (type.equals(node.path("type").asText())) return true;
        for (JsonNode child : node) if (containsType(child, type)) return true;
        return false;
    }
}
