/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class TokenizerFactory {

    public static Tokenizer load(Path modelDir, Path nativeLibPath) throws IOException {
        if (modelDir == null) {
            throw new IOException("Model path is null");
        }
        if (Files.isRegularFile(modelDir)) {
            Path parent = modelDir.getParent();
            if (parent == null) {
                throw new IOException("Model path has no parent directory: " + modelDir);
            }
            modelDir = parent;
        }

        Path hfConfig = modelDir.resolve("tokenizer.json");
        if (Files.exists(hfConfig)) {
            return loadHuggingFaceTokenizer(hfConfig);
        }

        Path subDir = modelDir.resolve("tokenizer");
        if (Files.isDirectory(subDir)) {
            Path subHf = subDir.resolve("tokenizer.json");
            if (Files.exists(subHf)) {
                return loadHuggingFaceTokenizer(subHf);
            }
        }

        throw new IOException("No supported tokenizer files found in " + modelDir + " (expected tokenizer.json)");
    }

    public static Tokenizer create(ModelConfig config) {
        if (config == null || config.getTokenizerPath() == null) {
            throw new IllegalArgumentException("ModelConfig or tokenizerPath is null");
        }
        return loadHuggingFaceTokenizer(config.getTokenizerPath());
    }

    private static Tokenizer loadHuggingFaceTokenizer(Path path) {
        HuggingFaceGemmaBpeLoader gemmaLoader = new HuggingFaceGemmaBpeLoader();
        try {
            if (gemmaLoader.supports(path)) return gemmaLoader.load(path);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to inspect tokenizer pipeline: " + path, e);
        }
        return new HuggingFaceBpeLoader().load(path);
    }
}
