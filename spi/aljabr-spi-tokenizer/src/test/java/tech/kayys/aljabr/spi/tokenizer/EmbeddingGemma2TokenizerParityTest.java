/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Opt-in integration parity test against a downloaded official EmbeddingGemma 2 tokenizer.json.
 *
 * <p>Set {@code -Daljabr.embeddinggemma2.tokenizer=/path/to/tokenizer.json}.
 * Expected IDs are pinned from QuixiAI/embeddinggemma.c's independent
 * EmbeddingGemma-2 GGUF tokenizer goldens, including its upstream HF revision.</p>
 */
class EmbeddingGemma2TokenizerParityTest {
    private static final String TOKENIZER_PROPERTY = "aljabr.embeddinggemma2.tokenizer";
    private static final String FIXTURE =
            "/tech/kayys/aljabr/spi/tokenizer/embeddinggemma2-tokenizer-reference.json";

    @Test
    @EnabledIfSystemProperty(named = TOKENIZER_PROPERTY, matches = ".+")
    void matchesPinnedEmbeddingGemma2ReferenceTokenIds() throws Exception {
        Path tokenizerPath = Path.of(System.getProperty(TOKENIZER_PROPERTY));
        assertTrue(Files.isRegularFile(tokenizerPath),
                "Expected tokenizer.json at " + tokenizerPath);

        JsonNode fixture;
        try (var input = getClass().getResourceAsStream(FIXTURE)) {
            assertTrue(input != null, "Missing pinned tokenizer reference fixture");
            fixture = new ObjectMapper().readTree(input);
        }

        Tokenizer tokenizer = new HuggingFaceGemmaBpeLoader().load(tokenizerPath);
        EncodeOptions options = EncodeOptions.builder().addBos(true).addEos(true).build();
        for (JsonNode testCase : fixture.path("cases")) {
            String text = testCase.path("text").asText();
            long[] expected = StreamSupport.stream(testCase.path("token_ids").spliterator(), false)
                    .mapToLong(JsonNode::asLong)
                    .toArray();
            assertArrayEquals(expected, tokenizer.encode(text, options),
                    "Token ID mismatch for input: " + abbreviate(text));
        }
    }

    private static String abbreviate(String text) {
        return text.length() <= 120 ? text : text.substring(0, 117) + "...";
    }
}
