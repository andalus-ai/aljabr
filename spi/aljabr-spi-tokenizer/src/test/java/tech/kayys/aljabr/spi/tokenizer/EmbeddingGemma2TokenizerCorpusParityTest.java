/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Full-corpus parity gate for the shared Gemma BPE tokenizer.
 *
 * <p>The upstream corpus is intentionally not copied into this repository: it contains
 * 608,225 expected IDs. Download the exact pinned artifacts and run this test with:
 *
 * <pre>
 * hf download google/embeddinggemma-2 tokenizer.json
 * hf download EverPaper/embeddinggemma-2-coreai tests/tokenizer-cases.json
 * ./gradlew :spi:aljabr-spi-tokenizer:test \
 *   -Daljabr.embeddinggemma2.tokenizer=/path/to/tokenizer.json \
 *   -Daljabr.embeddinggemma2.tokenizerCorpus=/path/to/tokenizer-cases.json
 * </pre>
 *
 * <p>The corpus contains raw, search-query, and document cases. Expected IDs were
 * generated with Hugging Face tokenizers 0.23.3 and are pinned by SHA-256 below.</p>
 */
class EmbeddingGemma2TokenizerCorpusParityTest {
    private static final String TOKENIZER_PROPERTY = "aljabr.embeddinggemma2.tokenizer";
    private static final String CORPUS_PROPERTY = "aljabr.embeddinggemma2.tokenizerCorpus";

    private static final String TOKENIZER_SHA256 =
            "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4";
    private static final String CORPUS_SHA256 =
            "0797f417a0fa96270dc2f439478000fe3b3a235eb1e83b3054cf69674e323ae9";

    private static final int EXPECTED_CASES = 3_758;
    private static final long EXPECTED_TOKEN_IDS = 608_225L;

    private static final Map<String, String> PREFIXES = Map.of(
            "raw", "",
            "q", "task: search result | query: ",
            "d", "title: none | text: ");

    @Test
    @EnabledIfSystemProperty(named = CORPUS_PROPERTY, matches = ".+")
    void matchesAllPinnedEmbeddingGemma2TokenizerCases() throws Exception {
        String tokenizerValue = System.getProperty(TOKENIZER_PROPERTY);
        assertTrue(tokenizerValue != null && !tokenizerValue.isBlank(),
                "Set -D" + TOKENIZER_PROPERTY + " to the pinned tokenizer.json");

        Path tokenizerPath = Path.of(tokenizerValue);
        Path corpusPath = Path.of(System.getProperty(CORPUS_PROPERTY));
        assertTrue(Files.isRegularFile(tokenizerPath), "Missing tokenizer.json: " + tokenizerPath);
        assertTrue(Files.isRegularFile(corpusPath), "Missing tokenizer corpus: " + corpusPath);
        assertSha256(tokenizerPath, TOKENIZER_SHA256, "tokenizer.json");
        assertSha256(corpusPath, CORPUS_SHA256, "tokenizer-cases.json");

        JsonNode cases;
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream input = Files.newInputStream(corpusPath)) {
            cases = mapper.readTree(input);
        }
        assertTrue(cases.isArray(), "Tokenizer corpus must be a JSON array");
        assertEquals(EXPECTED_CASES, cases.size(), "Unexpected tokenizer corpus case count");

        Tokenizer tokenizer = new HuggingFaceGemmaBpeLoader().load(tokenizerPath);
        EncodeOptions options = EncodeOptions.builder().addBos(true).addEos(true).build();

        long comparedTokenIds = 0;
        int mismatches = 0;
        StringBuilder diagnostics = new StringBuilder();
        for (int index = 0; index < cases.size(); index++) {
            JsonNode testCase = cases.get(index);
            String text = requiredText(testCase, "text", index);
            String kind = requiredText(testCase, "kind", index);
            String prefix = PREFIXES.get(kind);
            assertTrue(prefix != null, "Unknown case kind '" + kind + "' at index " + index);

            JsonNode ids = testCase.path("ids");
            assertTrue(ids.isArray(), "Missing expected ID array at index " + index);
            long[] expected = new long[ids.size()];
            for (int i = 0; i < ids.size(); i++) {
                assertTrue(ids.get(i).canConvertToLong(),
                        "Invalid token ID at case " + index + ", token " + i);
                expected[i] = ids.get(i).longValue();
            }

            long[] actual = tokenizer.encode(prefix + text, options);
            comparedTokenIds += expected.length;
            if (!java.util.Arrays.equals(expected, actual)) {
                mismatches++;
                if (mismatches <= 10) {
                    diagnostics.append("\ncase ").append(index)
                            .append(" (kind=").append(kind)
                            .append(", expected=").append(expected.length)
                            .append(", actual=").append(actual.length)
                            .append("): ").append(abbreviate(text));
                    diagnostics.append("\n  expected: ").append(abbreviateIds(expected));
                    diagnostics.append("\n  actual:   ").append(abbreviateIds(actual));
                }
            }
        }

        assertEquals(EXPECTED_TOKEN_IDS, comparedTokenIds,
                "Unexpected number of reference token IDs");
        assertEquals(0, mismatches,
                "Gemma tokenizer differs from the pinned Hugging Face tokenizers corpus; "
                        + mismatches + " of " + EXPECTED_CASES + " cases mismatched"
                        + diagnostics);
    }

    private static String requiredText(JsonNode node, String field, int index) {
        JsonNode value = node.path(field);
        assertTrue(value.isTextual(), "Missing textual field '" + field + "' at index " + index);
        return value.textValue();
    }

    private static void assertSha256(Path path, String expected, String description) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        String actual = HexFormat.of().formatHex(digest.digest());
        assertEquals(expected, actual, description + " SHA-256 mismatch");
    }

    private static String abbreviate(String value) {
        String visible = value.replace("\r", "\\r").replace("\n", "\\n");
        return visible.length() <= 100 ? visible : visible.substring(0, 97) + "...";
    }

    private static String abbreviateIds(long[] ids) {
        int limit = Math.min(ids.length, 24);
        String result = java.util.Arrays.toString(java.util.Arrays.copyOf(ids, limit));
        return ids.length <= limit ? result : result.substring(0, result.length() - 1) + ", ...]";
    }
}
