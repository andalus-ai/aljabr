package tech.kayys.aljabr.gguf.llamacpp;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LlamaCppProviderSettingsTest {

    @Test
    void defaultsAreSafeForCpuOnlyExecution() {
        LlamaCppProviderSettings settings = LlamaCppProviderSettings.defaults();
        assertFalse(settings.gpuEnabled());
        assertEquals(0, settings.gpuLayers());
        assertTrue(settings.mmapEnabled());
        assertEquals(4096, settings.maxContextTokens());
        assertEquals(Duration.ofSeconds(30), settings.defaultTimeout());
    }

    @Test
    void mapsRuntimeConfigurationOverrides() {
        LlamaCppProviderSettings settings = LlamaCppProviderSettings.from(Map.ofEntries(
                Map.entry("gpuEnabled", true),
                Map.entry("nGpuLayers", -1),
                Map.entry("nThreads", 6),
                Map.entry("nCtx", 8192),
                Map.entry("nBatch", 256),
                Map.entry("useMmap", false),
                Map.entry("coalesceEnabled", true),
                Map.entry("coalesceWindowMs", 5),
                Map.entry("coalesceMaxBatch", 12),
                Map.entry("coalesceMaxQueue", 96),
                Map.entry("maxConcurrentRequests", 4),
                Map.entry("timeout", "PT45S")));

        assertTrue(settings.gpuEnabled());
        assertEquals(-1, settings.gpuLayers());
        assertEquals(6, settings.threads());
        assertEquals(8192, settings.maxContextTokens());
        assertEquals(256, settings.batchSize());
        assertFalse(settings.mmapEnabled());
        assertTrue(settings.coalesceEnabled());
        assertEquals(5, settings.coalesceWindowMs());
        assertEquals(12, settings.coalesceMaxBatch());
        assertEquals(96, settings.coalesceMaxQueue());
        assertEquals(4, settings.maxConcurrentRequests());
        assertEquals(Duration.ofSeconds(45), settings.defaultTimeout());
    }
}
