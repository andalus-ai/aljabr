/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceGemmaBpeLoaderTest {
    @TempDir Path tempDir;

    @Test
    void loadsGemmaPipelineAndAppliesWhitespaceAndMergeRules() throws Exception {
        Tokenizer tokenizer = load("""
                {
                  "normalizer":{"type":"Sequence","normalizers":[
                    {"type":"Replace","pattern":{"String":" "},"content":"▁"}]},
                  "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious"},
                  "decoder":{"type":"Sequence","decoders":[
                    {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                    {"type":"ByteFallback"},{"type":"Fuse"}]},
                  "model":{"type":"BPE","byte_fallback":true,"fuse_unk":true,
                    "vocab":{"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3,
                      "h":4,"i":5,"▁":6,"t":7,"he":8,"▁t":9,"<0xE2>":10,"<0x98>":11,"<0x83>":12},
                    "merges":["h e","▁ t"]},
                  "added_tokens":[{"id":1,"content":"<bos>","special":true}]
                }
                """);

        long[] encoded = tokenizer.encode("he t", EncodeOptions.defaultOptions());
        assertArrayEquals(new long[]{8, 9}, encoded);
        assertEquals("he t", tokenizer.decode(encoded, DecodeOptions.defaultOptions()));
    }

    @Test
    void byteFallbackRoundTripsUtf8AndHonorsSpecialTokenSkipping() throws Exception {
        Tokenizer tokenizer = load("""
                {
                  "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
                  "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious"},
                  "decoder":{"type":"Sequence","decoders":[
                    {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                    {"type":"ByteFallback"},{"type":"Fuse"}]},
                  "model":{"type":"BPE","byte_fallback":true,"fuse_unk":true,
                    "vocab":{"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3,
                      "<0xE2>":4,"<0x98>":5,"<0x83>":6}},
                  "added_tokens":[{"id":1,"content":"<bos>","special":true}]
                }
                """);

        long[] encoded = tokenizer.encode("☃", EncodeOptions.defaultOptions());
        assertArrayEquals(new long[]{4, 5, 6}, encoded);
        assertEquals("☃", tokenizer.decode(encoded, DecodeOptions.defaultOptions()));
        assertEquals("", tokenizer.decode(new long[]{1}, DecodeOptions.defaultOptions()));
        assertEquals("<bos>", tokenizer.decode(new long[]{1},
                DecodeOptions.builder().skipSpecialTokens(false).build()));
    }

    @Test
    void rejectsNonGemmaPipelineInsteadOfSilentlyUsingWrongSemantics() throws Exception {
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, """
                {"model":{"type":"BPE","byte_fallback":false,"fuse_unk":true,"vocab":{"x":0}},
                 "normalizer":null,"pre_tokenizer":null,"decoder":null}
                """);
        assertThrows(IllegalArgumentException.class,
                () -> new HuggingFaceGemmaBpeLoader().load(file));
    }


    @Test
    void tokenizerFactorySelectsGemmaPipelineAutomatically() throws Exception {
        Path modelDir = tempDir.resolve("model");
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("tokenizer.json"), """
                {
                  "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
                  "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious"},
                  "decoder":{"type":"Sequence","decoders":[
                    {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                    {"type":"ByteFallback"},{"type":"Fuse"}]},
                  "model":{"type":"BPE","byte_fallback":true,"fuse_unk":true,
                    "vocab":{"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3,"h":4,"i":5,"▁":6},
                    "merges":[]}
                }
                """);
        Tokenizer tokenizer = TokenizerFactory.load(modelDir, null);
        assertArrayEquals(new long[]{4, 5, 6, 4}, tokenizer.encode("hi h", EncodeOptions.defaultOptions()));
    }

    private Tokenizer load(String json) throws Exception {
        Path file = tempDir.resolve("tokenizer-" + System.nanoTime() + ".json");
        Files.writeString(file, json);
        return new HuggingFaceGemmaBpeLoader().load(file);
    }
}
