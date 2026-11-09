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
        LlamaCppProviderSettings settings = LlamaCppProviderSettings.from(Map.of(
                "gpuEnabled", true,
                "nGpuLayers", -1,
                "nThreads", 6,
                "nCtx", 8192,
                "nBatch", 256,
                "useMmap", false,
                "coalesceEnabled", true,
                "coalesceWindowMs", 5,
                "coalesceMaxBatch", 12,
                "coalesceMaxQueue", 96,
                "maxConcurrentRequests", 4,
                "timeout", "PT45S"));

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
