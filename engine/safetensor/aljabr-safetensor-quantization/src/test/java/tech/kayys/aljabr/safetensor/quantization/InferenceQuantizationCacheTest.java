/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class InferenceQuantizationCacheTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void cacheIsDisabledForNullModelAndNoneStrategy() {
        InferenceQuantizationCache cache = new InferenceQuantizationCache();

        assertNull(cache.cachePath(null, QuantizationStrategy.INT4));
        assertNull(cache.cachePath(temporaryDirectory, QuantizationStrategy.NONE));
        assertNull(cache.cachePath(temporaryDirectory, null));
    }

    @Test
    void cachePathUsesConfiguredAljabrHome() {
        String previous = System.getProperty("aljabr.home");
        System.setProperty("aljabr.home", temporaryDirectory.toString());
        try {
            InferenceQuantizationCache cache = new InferenceQuantizationCache();
            Path path = cache.cachePath(temporaryDirectory.resolve("model"), QuantizationStrategy.INT4);

            assertNotNull(path);
            assertTrue(path.startsWith(temporaryDirectory.resolve("cache").resolve("direct-quant")));
            assertTrue(path.toString().endsWith(".dqcache"));
        } finally {
            if (previous == null) {
                System.clearProperty("aljabr.home");
            } else {
                System.setProperty("aljabr.home", previous);
            }
        }
    }

    @Test
    void missingAndUnsupportedVersionCachesAreTreatedAsMisses() throws Exception {
        InferenceQuantizationCache cache = new InferenceQuantizationCache();
        Path missing = temporaryDirectory.resolve("missing.dqcache");
        assertTrue(cache.load(missing, QuantizationStrategy.INT4).isEmpty());

        Path unsupported = temporaryDirectory.resolve("unsupported.dqcache");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(unsupported))) {
            out.writeInt(999);
        }
        assertTrue(cache.load(unsupported, QuantizationStrategy.INT4).isEmpty());
    }

    @Test
    void strategyMismatchIsTreatedAsCacheMiss() throws Exception {
        InferenceQuantizationCache cache = new InferenceQuantizationCache();
        Path path = temporaryDirectory.resolve("strategy-mismatch.dqcache");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(path))) {
            out.writeInt(1);
            out.writeUTF(QuantizationStrategy.INT8.name());
        }

        assertTrue(cache.load(path, QuantizationStrategy.INT4).isEmpty());
    }
}
