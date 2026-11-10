package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class GgufPromptCacheDiskStorageTest {

    @Test
    void testSaveAndLoadPromptCache(@TempDir Path tempDir) throws IOException {
        Path fakeModel = tempDir.resolve("test_model.gguf");
        Files.writeString(fakeModel, "dummy-model-data");

        int[] tokens = new int[] {106, 1645, 108, 4794, 235338, 107};
        int nLayers = 4;
        int T = tokens.length;
        int stride = 64;

        float[][] k = new float[nLayers][T * stride];
        float[][] v = new float[nLayers][T * stride];
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < T * stride; i++) {
                k[l][i] = (float) (l * 1000 + i * 0.1);
                v[l][i] = (float) (l * 2000 - i * 0.2);
            }
        }
        float[] logits = new float[] {0.1f, -0.5f, 12.3f, 4.2f};
        PromptCacheEntry original = new PromptCacheEntry(tokens, k, v, logits);

        Path cacheDir = tempDir.resolve("cache");
        GgufPromptCacheDiskStorage.saveToDisk(cacheDir, fakeModel, original);

        PromptCacheEntry loaded = GgufPromptCacheDiskStorage.loadFromDisk(cacheDir, fakeModel, tokens);
        assertNotNull(loaded, "Loaded entry should not be null");
        assertArrayEquals(tokens, loaded.tokens);
        assertArrayEquals(logits, loaded.logits, 1e-6f);
        assertEquals(nLayers, loaded.k.length);
        assertEquals(nLayers, loaded.v.length);
        for (int l = 0; l < nLayers; l++) {
            assertArrayEquals(k[l], loaded.k[l], 1e-6f);
            assertArrayEquals(v[l], loaded.v[l], 1e-6f);
        }

        // Mismatched tokens should miss
        int[] otherTokens = new int[] {106, 1645, 108, 4794, 235338, 999};
        PromptCacheEntry miss = GgufPromptCacheDiskStorage.loadFromDisk(cacheDir, fakeModel, otherTokens);
        assertNull(miss, "Different tokens should return null (miss)");

        // Non-existent model should miss
        Path otherModel = tempDir.resolve("other_model.gguf");
        Files.writeString(otherModel, "other-model-data");
        PromptCacheEntry modelMiss = GgufPromptCacheDiskStorage.loadFromDisk(cacheDir, otherModel, tokens);
        assertNull(modelMiss, "Different model should return null");
    }

    @Test
    void testCorruptFileReturnsNull(@TempDir Path tempDir) throws IOException {
        Path fakeModel = tempDir.resolve("test_model.gguf");
        Files.writeString(fakeModel, "dummy-model-data");
        int[] tokens = new int[] {1, 2, 3};

        Path cacheDir = tempDir.resolve("cache");
        Files.createDirectories(cacheDir);
        String key = GgufPromptCacheDiskStorage.computeKey(fakeModel, tokens);
        Path targetFile = cacheDir.resolve(key + ".pkv");
        Files.write(targetFile, new byte[] {0, 1, 2, 3}); // corrupt

        PromptCacheEntry loaded = GgufPromptCacheDiskStorage.loadFromDisk(cacheDir, fakeModel, tokens);
        assertNull(loaded, "Corrupt file should safely return null");
    }
}
