package tech.kayys.aljabr.gguf.llamacpp;

import java.time.Duration;
import java.util.Map;

/**
 * Plain Java runtime settings for the llama.cpp GGUF provider.
 *
 * <p>This type is intentionally independent of Quarkus/SmallRye Config so
 * ServiceLoader-created backends can be used from CLI, tests, and non-Quarkus
 * applications. Framework integrations translate their external configuration
 * into this value at the boundary.</p>
 */
public record LlamaCppProviderSettings(
        boolean verboseLogging,
        boolean gpuEnabled,
        int gpuLayers,
        int threads,
        int maxContextTokens,
        int batchSize,
        boolean mmapEnabled,
        boolean mlockEnabled,
        int gpuDeviceId,
        int coalesceSeqMax,
        boolean coalesceEnabled,
        int coalesceWindowMs,
        int coalesceMaxBatch,
        int coalesceMaxQueue,
        int maxConcurrentRequests,
        Duration defaultTimeout) {

    public LlamaCppProviderSettings {
        threads = Math.max(1, threads);
        maxContextTokens = Math.max(512, maxContextTokens);
        batchSize = Math.max(1, batchSize);
        coalesceSeqMax = Math.max(1, coalesceSeqMax);
        coalesceWindowMs = Math.max(0, coalesceWindowMs);
        coalesceMaxBatch = Math.max(1, coalesceMaxBatch);
        coalesceMaxQueue = Math.max(1, coalesceMaxQueue);
        maxConcurrentRequests = Math.max(1, maxConcurrentRequests);
        defaultTimeout = defaultTimeout == null ? Duration.ofSeconds(30) : defaultTimeout;
    }

    public static LlamaCppProviderSettings defaults() {
        return new LlamaCppProviderSettings(
                false, false, 0,
                Runtime.getRuntime().availableProcessors(),
                4096, 128, true, false, 0, 1,
                false, 3, 8, 64, 10, Duration.ofSeconds(30));
    }

    public static LlamaCppProviderSettings from(Map<String, Object> values) {
        Map<String, Object> config = values == null ? Map.of() : values;
        LlamaCppProviderSettings defaults = defaults();
        return new LlamaCppProviderSettings(
                bool(config, "verboseLogging", defaults.verboseLogging()),
                bool(config, "gpuEnabled", defaults.gpuEnabled()),
                integer(config, defaults.gpuLayers(), "nGpuLayers", "gpuLayers"),
                integer(config, defaults.threads(), "nThreads", "threads"),
                integer(config, defaults.maxContextTokens(), "nCtx", "maxContextTokens"),
                integer(config, defaults.batchSize(), "nBatch", "batchSize"),
                bool(config, "useMmap", defaults.mmapEnabled()),
                bool(config, "useMlock", defaults.mlockEnabled()),
                integer(config, defaults.gpuDeviceId(), "gpuDeviceId"),
                integer(config, defaults.coalesceSeqMax(), "coalesceSeqMax"),
                bool(config, "coalesceEnabled", defaults.coalesceEnabled()),
                integer(config, defaults.coalesceWindowMs(), "coalesceWindowMs"),
                integer(config, defaults.coalesceMaxBatch(), "coalesceMaxBatch"),
                integer(config, defaults.coalesceMaxQueue(), "coalesceMaxQueue"),
                integer(config, defaults.maxConcurrentRequests(), "maxConcurrentRequests"),
                duration(config.get("timeout"), defaults.defaultTimeout()));
    }

    private static boolean bool(Map<String, Object> values, String key, boolean fallback) {
        Object value = values.get(key);
        if (value instanceof Boolean bool) return bool;
        if (value instanceof String text) return Boolean.parseBoolean(text.trim());
        return fallback;
    }

    private static int integer(Map<String, Object> values, int fallback, String... keys) {
        for (String key : keys) {
            Object value = values.get(key);
            if (value instanceof Number number) return number.intValue();
            if (value instanceof String text) {
                try { return Integer.parseInt(text.trim()); }
                catch (NumberFormatException ignored) { }
            }
        }
        return fallback;
    }

    private static Duration duration(Object value, Duration fallback) {
        if (value instanceof Duration duration) return duration;
        if (value instanceof String text) {
            try { return Duration.parse(text.trim()); }
            catch (RuntimeException ignored) { }
        }
        return fallback;
    }
}
