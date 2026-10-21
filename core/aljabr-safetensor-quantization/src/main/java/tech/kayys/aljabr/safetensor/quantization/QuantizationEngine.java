/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizationEngine.java
 * ───────────────────────
 * Model quantization support (GPTQ, FP8).
 */
package tech.kayys.aljabr.safetensor.quantization;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.aljabr.safetensor.quantization.bridge.AccelSafetensorWriter;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.safetensor.quantization.quantizer.*;
import tech.kayys.aljabr.safetensor.quantization.quantizer.TurboQuantAdapter;
import tech.kayys.aljabr.safetensor.quantization.quantizer.BnBQuantizerAdapter;
import tech.kayys.aljabr.safetensor.quantization.quantizer.AWQQuantizerAdapter;
import tech.kayys.aljabr.safetensor.quantization.quantizer.GPTQQuantizerAdapter;

import java.io.IOException;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Quantization engine for model compression.
 * <p>
 * Supports multiple quantization strategies:
 * <ul>
 * <li>INT4 - 4-bit integer quantization using GPTQ algorithm</li>
 * <li>INT8 - 8-bit integer quantization with per-channel scaling</li>
 * <li>FP8 - 8-bit floating point quantization (E4M3/E5M2)</li>
 * </ul>
 *
 * @author bhangun
 * @version 1.0.0
 */
@ApplicationScoped
public class QuantizationEngine {

    private static final Logger log = Logger.getLogger(QuantizationEngine.class);

    /**
     * Quantization strategy.
     */
    public enum QuantStrategy {
        NONE,
        INT4,
        INT8,
        FP8,
        TURBO,
        AWQ,
        GPTQ,
        BNB
    }

    /**
     * Quantization progress information.
     */
    public record QuantProgress(
            String tensorName,
            int currentTensor,
            int totalTensors,
            double percentComplete,
            String phase,
            String message) {
    }

    public record InferenceQuantizationResult(
            Map<String, AccelTensor> weights,
            String cacheState,
            Path cachePath,
            int cacheHits,
            int quantizedFresh) {
    }

    private final Map<String, Quantizer> quantizers = new ConcurrentHashMap<>();
    private volatile ExecutorService executor;
    private final Map<Path, QuantResult> resultsCache = new ConcurrentHashMap<>();
    private static final int INFERENCE_CACHE_VERSION = 1;

    @Inject
    SafetensorShardLoader loader;

    @Inject
    AccelWeightBridge weightBridge;

    /**
     * Initialize quantization engine with default quantizers.
     */
    public QuantizationEngine() {
        // Register legacy/generic placeholders
        registerQuantizer(QuantStrategy.INT4, new GPTQQuantizerAdapter());
        registerQuantizer(QuantStrategy.INT8, new BnBQuantizerAdapter()); // INT8 fallback
        
        // Register high-performance adapters from core/quantizer
        registerQuantizer(QuantStrategy.TURBO, new TurboQuantAdapter());
        registerQuantizer(QuantStrategy.AWQ, new AWQQuantizerAdapter());
        registerQuantizer(QuantStrategy.BNB, new BnBQuantizerAdapter());
        registerQuantizer(QuantStrategy.INT4, new GPTQQuantizerAdapter());
        registerQuantizer(QuantStrategy.INT4, new AutoRoundQuantizerAdapter());
        
        log.info("QuantizationEngine initialized with TurboQuant, AWQ, GPTQ, AutoRound, and BnB adapters.");
    }

    /**
     * Register a custom quantizer.
     *
     * @param strategy strategy this quantizer handles
     * @param quantizer quantizer implementation
     */
    public void registerQuantizer(QuantStrategy strategy, Quantizer quantizer) {
        quantizers.put(strategy.name().toLowerCase(), quantizer);
        log.infof("Registered quantizer for strategy: %s", strategy);
    }

    /**
     * Quantize a model asynchronously.
     *
     * @param modelPath     path to the model directory
     * @param outputPath    path for the quantized output
     * @param strategy      quantization strategy
     * @param config        quantization configuration
     * @return Uni emitting quantization result
     */
    public Uni<QuantResult> quantizeAsync(Path modelPath, Path outputPath, QuantStrategy strategy, QuantConfig config) {
        return Uni.createFrom().<QuantResult>emitter(em -> {
            try {
                QuantResult result = quantize(modelPath, outputPath, strategy, config);
                em.complete(result);
            } catch (Exception e) {
                em.fail(e);
            }
        }).runSubscriptionOn(executor());
    }

    /**
     * Quantize a model with progress streaming.
     *
     * @param modelPath     path to the model directory
     * @param outputPath    path for the quantized output
     * @param strategy      quantization strategy
     * @param config        quantization configuration
     * @return Multi emitting progress updates and final result
     */
    public Multi<Object> quantizeWithProgress(Path modelPath, Path outputPath, QuantStrategy strategy,
            QuantConfig config) {
        return Multi.createFrom().emitter(em -> {
            Instant startTime = Instant.now();

            try {
                // Validate input
                validateModelPath(modelPath);

                // Create output directory
                Files.createDirectories(outputPath);

                // Get quantizer
                Quantizer quantizer = getQuantizer(strategy);
                if (quantizer == null) {
                    throw new IllegalStateException("No quantizer found for strategy: " + strategy);
                }

                // Load model weights
                em.emit(new QuantProgress("", 0, 0, 0.0, "LOADING", "Loading model weights..."));
                Map<String, AccelTensor> weights = loadWeights(modelPath);
                long originalSize = calculateTotalSize(weights);

                em.emit(new QuantProgress("", 0, weights.size(), 0.0, "PREPARING",
                        "Preparing for quantization..."));

                // Perform quantization with progress
                Map<String, AccelTensor> quantizedWeights = new HashMap<>();
                int totalTensors = weights.size();
                int currentTensor = 0;

                for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
                    String tensorName = entry.getKey();
                    AccelTensor tensor = entry.getValue();

                    em.emit(new QuantProgress(
                            tensorName,
                            currentTensor,
                            totalTensors,
                            (double) currentTensor / totalTensors * 100.0,
                            "QUANTIZING",
                            String.format("Quantizing tensor %d/%d: %s", currentTensor + 1, totalTensors, tensorName)));

                    AccelTensor quantizedTensor = quantizer.quantizeTensor(tensor, config);
                    quantizedWeights.put(tensorName, quantizedTensor);

                    currentTensor++;
                }

                em.emit(new QuantProgress("", totalTensors, totalTensors, 100.0, "SAVING", "Saving quantized model..."));

                // Save quantized weights
                saveWeights(outputPath, quantizedWeights);

                // Calculate statistics
                long quantizedSize = calculateTotalSize(quantizedWeights);
                QuantStats stats = QuantStats.builder()
                        .originalSizeBytes(originalSize)
                        .quantizedSizeBytes(quantizedSize)
                        .tensorCount(totalTensors)
                        .paramCount(countParameters(weights))
                        .startTime(startTime)
                        .endTime(Instant.now())
                        .status(QuantStats.Status.COMPLETED)
                        .build();

                QuantResult result = QuantResult.success(outputPath, stats, config);
                resultsCache.put(outputPath, result);

                em.emit(result);
                em.complete();

            } catch (Exception e) {
                log.errorf(e, "Quantization failed");
                em.emit(QuantResult.failure(e.getMessage(), config));
                em.fail(e);
            }
        }).runSubscriptionOn(executor());
    }

    private ExecutorService executor() {
        ExecutorService current = executor;
        if (current != null) {
            return current;
        }
        synchronized (this) {
            if (executor == null) {
                executor = Executors.newVirtualThreadPerTaskExecutor();
            }
            return executor;
        }
    }

    /**
     * Quantize a model.
     *
     * @param modelPath  path to the model directory
     * @param outputPath path for the quantized output
     * @param strategy   quantization strategy
     * @return quantization result
     */
    public QuantResult quantize(Path modelPath, Path outputPath, QuantStrategy strategy) {
        return quantize(modelPath, outputPath, strategy, createDefaultConfig(strategy));
    }

    /**
     * Quantize a model with custom configuration.
     *
     * @param modelPath  path to the model directory
     * @param outputPath path for the quantized output
     * @param strategy   quantization strategy
     * @param config     quantization configuration
     * @return quantization result
     */
    public QuantResult quantize(Path modelPath, Path outputPath, QuantStrategy strategy, QuantConfig config) {
        log.infof("Starting quantization: %s -> %s (strategy=%s)", modelPath, outputPath, strategy);

        Instant startTime = Instant.now();
        try {
            // Validate input
            validateModelPath(modelPath);

            // Create output directory
            Files.createDirectories(outputPath);

            // Get quantizer
            Quantizer quantizer = getQuantizer(strategy);
            if (quantizer == null) {
                throw new IllegalStateException("No quantizer found for strategy: " + strategy);
            }

            // Load model weights
            log.infof("Loading model weights from: %s", modelPath);
            Map<String, AccelTensor> weights = loadWeights(modelPath);
            long originalSize = calculateTotalSize(weights);
            log.infof("Loaded %d tensors (%s)", weights.size(), QuantStats.formatSize(originalSize));

            // Perform quantization
            log.infof("Quantizing with strategy: %s", strategy);
            Map<String, AccelTensor> quantizedWeights = new HashMap<>();
            int totalTensors = weights.size();
            int currentTensor = 0;

            for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
                String tensorName = entry.getKey();
                AccelTensor tensor = entry.getValue();

                if (currentTensor % 10 == 0) {
                    log.infof("Progress: %d/%d tensors (%.1f%%)", currentTensor, totalTensors,
                            (double) currentTensor / totalTensors * 100.0);
                }

                AccelTensor quantizedTensor = quantizer.quantizeTensor(tensor, config);
                quantizedWeights.put(tensorName, quantizedTensor);
                tensor.close(); // Free original tensor memory

                currentTensor++;
            }

            // Save quantized weights
            log.infof("Saving quantized model to: %s", outputPath);
            saveWeights(outputPath, quantizedWeights);

            // Calculate statistics
            long quantizedSize = calculateTotalSize(quantizedWeights);
            QuantStats stats = QuantStats.builder()
                    .originalSizeBytes(originalSize)
                    .quantizedSizeBytes(quantizedSize)
                    .tensorCount(totalTensors)
                    .paramCount(countParameters(weights))
                    .startTime(startTime)
                    .endTime(Instant.now())
                    .status(QuantStats.Status.COMPLETED)
                    .build();

            QuantResult result = QuantResult.success(outputPath, stats, config);
            resultsCache.put(outputPath, result);

            log.infof("Quantization complete: %s -> %s", QuantStats.formatSize(originalSize),
                    QuantStats.formatSize(quantizedSize));
            log.infof("Compression ratio: %.2fx", stats.getCompressionRatio());

            return result;

        } catch (Exception e) {
            log.errorf(e, "Quantization failed");
            QuantStats stats = QuantStats.builder()
                    .startTime(startTime)
                    .endTime(Instant.now())
                    .status(QuantStats.Status.FAILED)
                    .errorMessage(e.getMessage())
                    .build();
            return QuantResult.failure(e.getMessage(), config);
        }
    }

    /**
     * Load a quantized model for inference.
     *
     * @param quantizedModelPath path to quantized model
     * @param strategy           quantization strategy
     * @return dequantized weights map
     */
    public Map<String, AccelTensor> loadQuantizedModel(Path quantizedModelPath, QuantStrategy strategy) {
        log.infof("Loading quantized model: %s (strategy=%s)", quantizedModelPath, strategy);

        try {
            Map<String, AccelTensor> quantizedWeights = loadWeights(quantizedModelPath);
            Quantizer quantizer = getQuantizer(strategy);

            if (quantizer == null) {
                throw new IllegalStateException("No quantizer found for strategy: " + strategy);
            }

            // Dequantize weights
            Map<String, AccelTensor> dequantizedWeights = new HashMap<>();
            for (Map.Entry<String, AccelTensor> entry : quantizedWeights.entrySet()) {
                AccelTensor dequantized = quantizer.dequantizeTensor(entry.getValue(), createDefaultConfig(strategy));
                dequantizedWeights.put(entry.getKey(), dequantized);
            }

            log.infof("Loaded %d quantized tensors", dequantizedWeights.size());
            return dequantizedWeights;

        } catch (IOException e) {
            throw new RuntimeException("Failed to load quantized model", e);
        }
    }

    /**
     * Get cached quantization result.
     *
     * @param modelPath path to quantized model
     * @return cached result or null
     */
    public QuantResult getCachedResult(Path modelPath) {
        return resultsCache.get(modelPath);
    }

    /**
     * Quantize eligible inference weights in memory.
     * Keeps small/vector tensors in their original form to reduce breakage risk.
     */
    public Map<String, AccelTensor> quantizeWeightsForInference(Map<String, AccelTensor> weights, QuantStrategy strategy) {
        return quantizeWeightsForInferenceDetailed(null, weights, strategy).weights();
    }

    public Map<String, AccelTensor> quantizeWeightsForInference(Path modelPath, Map<String, AccelTensor> weights, QuantStrategy strategy) {
        return quantizeWeightsForInferenceDetailed(modelPath, weights, strategy).weights();
    }

    public InferenceQuantizationResult quantizeWeightsForInferenceDetailed(Path modelPath, Map<String, AccelTensor> weights, QuantStrategy strategy) {
        QuantStrategy effectiveStrategy = normalizeInferenceStrategy(strategy);
        if (effectiveStrategy == QuantStrategy.NONE) {
            return new InferenceQuantizationResult(weights, "off", null, 0, 0);
        }

        Quantizer quantizer = getQuantizer(effectiveStrategy);
        if (quantizer == null) {
            log.warnf("No quantizer registered for strategy %s; using original weights", effectiveStrategy);
            return new InferenceQuantizationResult(weights, "unsupported", null, 0, 0);
        }

        QuantConfig config = createDefaultConfig(effectiveStrategy);
        Path cachePath = inferenceCachePath(modelPath, effectiveStrategy);
        Map<String, CachedTensor> cachedTensors = loadInferenceCache(cachePath, effectiveStrategy);

        // Separate tensors that need quantization from those that can pass through.
        Map<String, AccelTensor> passThrough = new ConcurrentHashMap<>();
        Map<String, AccelTensor> toQuantize  = new LinkedHashMap<>();
        for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
            if (!shouldQuantizeForInference(entry.getKey(), entry.getValue())) {
                passThrough.put(entry.getKey(), entry.getValue());
            } else {
                toQuantize.put(entry.getKey(), entry.getValue());
            }
        }

        // Check cache first (serial, cheap)
        AtomicInteger cacheHits = new AtomicInteger(0);
        Map<String, AccelTensor> fromCache = new ConcurrentHashMap<>();
        CopyOnWriteArrayList<Map.Entry<String, AccelTensor>> needsQuant = new CopyOnWriteArrayList<>();
        for (Map.Entry<String, AccelTensor> entry : toQuantize.entrySet()) {
            String name = entry.getKey();
            CachedTensor cached = cachedTensors.get(name);
            if (cached != null) {
                try {
                    fromCache.put(name, cached.toTensor());
                    entry.getValue().close();
                    cacheHits.incrementAndGet();
                    continue;
                } catch (Exception e) {
                    log.warnf(e, "Failed to hydrate cached tensor for %s; will re-quantize", name);
                }
            }
            needsQuant.add(entry);
        }

        int totalToQuant = needsQuant.size();
        if (totalToQuant > 0) {
            int cpuCores = Runtime.getRuntime().availableProcessors();
            System.err.printf("[Aljabr] Quantizing %d tensors (BnB NF4) using %d CPU threads...%n",
                    totalToQuant, cpuCores);
        }

        // Parallel quantization — uses all CPU cores via ForkJoinPool work-stealing.
        AtomicInteger doneCount = new AtomicInteger(0);
        AtomicInteger quantizedFresh = new AtomicInteger(0);
        Map<String, AccelTensor> freshQuant = new ConcurrentHashMap<>();

        ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        try {
            pool.submit(() -> needsQuant.parallelStream().forEach(entry -> {
                String name = entry.getKey();
                AccelTensor source = entry.getValue();
                int idx = doneCount.incrementAndGet();
                long t0 = System.currentTimeMillis();
                try {
                    AccelTensor materialized = materializeForQuantization(source);
                    AccelTensor candidate = quantizer.quantizeTensor(materialized, config);
                    long ms = System.currentTimeMillis() - t0;
                    if (!isUsableQuantizedTensor(candidate)) {
                        System.err.printf("[Aljabr] [%d/%d] SKIP (unsupported output) %s  (%.1fs)%n",
                                idx, totalToQuant, name, ms / 1000.0);
                        log.warnf("Quantizer %s produced unsupported tensor for %s", effectiveStrategy, name);
                        if (candidate != null && candidate != source && candidate != materialized) candidate.close();
                        freshQuant.put(name, source);
                        return;
                    }
                    System.err.printf("[Aljabr] [%d/%d] OK  %s  (%.1fs, %s → NF4)%n",
                            idx, totalToQuant, name, ms / 1000.0, source.quantType());
                    freshQuant.put(name, candidate);
                    quantizedFresh.incrementAndGet();
                    if (materialized != source) materialized.close();
                    source.close();
                } catch (Exception e) {
                    long ms = System.currentTimeMillis() - t0;
                    System.err.printf("[Aljabr] [%d/%d] ERR %s  (%.1fs): %s%n",
                            idx, totalToQuant, name, ms / 1000.0, e.getMessage());
                    log.warnf(e, "Failed to quantize %s; keeping original", name);
                    freshQuant.put(name, source);
                }
            })).get();
        } catch (Exception e) {
            log.errorf(e, "Parallel quantization pool failed");
        } finally {
            pool.shutdown();
        }

        // Merge results
        Map<String, AccelTensor> quantized = new HashMap<>(weights.size() * 2);
        quantized.putAll(passThrough);
        quantized.putAll(fromCache);
        quantized.putAll(freshQuant);

        if (cachePath != null && quantizedFresh.get() > 0) {
            persistInferenceCache(cachePath, effectiveStrategy, quantized);
        }

        if (cacheHits.get() > 0) {
            log.infof("Loaded %d quantized tensors from inference cache (%s)", cacheHits.get(), effectiveStrategy);
        }
        System.err.printf("[Aljabr] Quantization complete: %d fresh, %d cached, %d pass-through%n",
                quantizedFresh.get(), cacheHits.get(), passThrough.size());
        log.infof("Prepared %d inference weights with strategy %s", quantized.size(), effectiveStrategy);
        String cacheState = cacheHits.get() > 0 ? "warm" : (quantizedFresh.get() > 0 ? "cold" : "bypass");
        return new InferenceQuantizationResult(quantized, cacheState, cachePath, cacheHits.get(), quantizedFresh.get());
    }

    /**
     * Clear the results cache.
     */
    public void clearCache() {
        resultsCache.clear();
        log.info("Quantization results cache cleared");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal methods
    // ─────────────────────────────────────────────────────────────────────────

    private Quantizer getQuantizer(QuantStrategy strategy) {
        return quantizers.get(strategy.name().toLowerCase());
    }

    private QuantStrategy normalizeInferenceStrategy(QuantStrategy strategy) {
        if (strategy == null) {
            return QuantStrategy.NONE;
        }

        return switch (strategy) {
            case TURBO -> {
                log.warn("TurboQuant direct inference is not wired yet; using BNB-compatible NF4 path");
                yield QuantStrategy.BNB;
            }
            case GPTQ -> {
                log.warn("GPTQ direct inference path is not complete yet; using INT4-compatible path");
                yield QuantStrategy.INT4;
            }
            default -> strategy;
        };
    }

    private boolean shouldQuantizeForInference(String name, AccelTensor tensor) {
        return tensor != null
                && (name.endsWith(".weight") || name.endsWith(".qweight") || name.endsWith("down_proj") || name.endsWith("gate_up_proj"))
                && (tensor.rank() == 2 || tensor.rank() == 3)
                && tensor.numel() >= 4096
                && !name.contains("embed_tokens")
                && !name.contains("position_embedding")
                && !name.contains("vision_tower");
    }

    private Path inferenceCachePath(Path modelPath, QuantStrategy strategy) {
        if (modelPath == null || strategy == null || strategy == QuantStrategy.NONE) {
            return null;
        }
        try {
            String aljabrHome = System.getProperty("aljabr.home",
                    Path.of(System.getProperty("user.home"), ".aljabr").toString());
            Path cacheDir = ensureInferenceCacheDirectory(Path.of(aljabrHome, "cache", "direct-quant"));
            Path normalized = modelPath.toAbsolutePath().normalize();
            long lastModified = Files.exists(normalized) ? Files.getLastModifiedTime(normalized).toMillis() : 0L;
            long size = Files.isRegularFile(normalized) ? Files.size(normalized) : 0L;
            String cacheName = Integer.toHexString((normalized.toString() + "|" + strategy.name() + "|" + lastModified + "|" + size).hashCode());
            return cacheDir.resolve(cacheName + ".dqcache");
        } catch (IOException e) {
            log.warnf(e, "Failed to prepare inference cache path for %s", modelPath);
            return null;
        }
    }

    private Path ensureInferenceCacheDirectory(Path preferred) throws IOException {
        try {
            Files.createDirectories(preferred);
            return preferred;
        } catch (IOException preferredFailure) {
            Path fallback = Path.of(System.getProperty("java.io.tmpdir"), "aljabr-direct-quant");
            Files.createDirectories(fallback);
            log.warnf(preferredFailure, "Falling back to temp inference cache directory: %s", fallback);
            return fallback;
        }
    }

    private Map<String, CachedTensor> loadInferenceCache(Path cachePath, QuantStrategy strategy) {
        if (cachePath == null || !Files.isRegularFile(cachePath)) {
            return Collections.emptyMap();
        }
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(cachePath)))) {
            int version = in.readInt();
            if (version != INFERENCE_CACHE_VERSION) {
                log.warnf("Ignoring inference cache with unsupported version %d at %s", version, cachePath);
                return Collections.emptyMap();
            }
            String strategyName = in.readUTF();
            if (!strategy.name().equals(strategyName)) {
                return Collections.emptyMap();
            }
            int count = in.readInt();
            Map<String, CachedTensor> cached = new HashMap<>(count * 2);
            for (int i = 0; i < count; i++) {
                String name = in.readUTF();
                int rank = in.readInt();
                long[] shape = new long[rank];
                for (int j = 0; j < rank; j++) {
                    shape[j] = in.readLong();
                }
                int quantOrdinal = in.readInt();
                int groupSize = in.readInt();
                int dataLength = in.readInt();
                byte[] data = in.readNBytes(dataLength);
                int scalesLength = in.readInt();
                float[] scales = new float[scalesLength];
                for (int j = 0; j < scalesLength; j++) {
                    scales[j] = in.readFloat();
                }
                int zerosLength = in.readInt();
                float[] zeros = new float[zerosLength];
                for (int j = 0; j < zerosLength; j++) {
                    zeros[j] = in.readFloat();
                }
                cached.put(name, new CachedTensor(shape,
                        AccelTensor.QuantType.values()[quantOrdinal],
                        groupSize, data, scales, zeros));
            }
            log.infof("Loaded inference quant cache: %s (%d tensors)", cachePath, count);
            return cached;
        } catch (Exception e) {
            log.warnf(e, "Failed to load inference quant cache from %s", cachePath);
            return Collections.emptyMap();
        }
    }

    private void persistInferenceCache(Path cachePath, QuantStrategy strategy, Map<String, AccelTensor> weights) {
        if (cachePath == null) {
            return;
        }
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(cachePath)))) {
            List<Map.Entry<String, AccelTensor>> entries = weights.entrySet().stream()
                    .filter(e -> isPersistableQuantizedTensor(e.getValue()))
                    .sorted(Map.Entry.comparingByKey())
                    .toList();
            out.writeInt(INFERENCE_CACHE_VERSION);
            out.writeUTF(strategy.name());
            out.writeInt(entries.size());
            for (Map.Entry<String, AccelTensor> entry : entries) {
                AccelTensor tensor = entry.getValue();
                out.writeUTF(entry.getKey());
                out.writeInt(tensor.rank());
                for (long dim : tensor.shape()) {
                    out.writeLong(dim);
                }
                out.writeInt(tensor.quantType().ordinal());
                out.writeInt(tensor.groupSize());
                byte[] data = tensor.dataSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                out.writeInt(data.length);
                out.write(data);
                float[] scales = tensor.scales() != null
                        ? tensor.scales().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT)
                        : new float[0];
                out.writeInt(scales.length);
                for (float scale : scales) {
                    out.writeFloat(scale);
                }
                float[] zeros = tensor.zeros() != null
                        ? tensor.zeros().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT)
                        : new float[0];
                out.writeInt(zeros.length);
                for (float zero : zeros) {
                    out.writeFloat(zero);
                }
            }
            log.infof("Saved inference quant cache: %s (%d tensors)", cachePath, entries.size());
        } catch (Exception e) {
            log.warnf(e, "Failed to persist inference quant cache to %s", cachePath);
        }
    }

    private boolean isPersistableQuantizedTensor(AccelTensor tensor) {
        return tensor != null
                && tensor.quantType() == AccelTensor.QuantType.INT4
                && tensor.scales() != null;
    }

    private record CachedTensor(long[] shape, AccelTensor.QuantType quantType, int groupSize, byte[] data, float[] scales, float[] zeros) {
        private AccelTensor toTensor() {
            AccelTensor tensor = AccelTensor.fromByteArray(data, shape);
            java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
            java.lang.foreign.MemorySegment scaleSeg = scales.length == 0
                    ? null
                    : arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_FLOAT, scales);
            java.lang.foreign.MemorySegment zeroSeg = zeros.length == 0
                    ? null
                    : arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_FLOAT, zeros);
            return tensor.withQuantization(quantType, scaleSeg, zeroSeg, groupSize);
        }
    }

    private AccelTensor materializeForQuantization(AccelTensor tensor) {
        if (tensor.quantType() == AccelTensor.QuantType.F32) {
            return tensor;
        }
        // BF16 and F16 tensors are handled directly by BnBQuantizerAdapter via chunked
        // MemorySegment reads — avoid allocating a full F32 copy here, which would OOM
        // for large tensors like lm_head (~2.5GB for a 35B MoE model).
        if (tensor.quantType() == AccelTensor.QuantType.BF16
                || tensor.quantType() == AccelTensor.QuantType.F16) {
            return tensor;
        }
        return tensor.dequantize();
    }


    private boolean isUsableQuantizedTensor(AccelTensor tensor) {
        if (tensor == null) {
            return false;
        }

        return switch (tensor.quantType()) {
            case F32, F16, BF16 -> true;
            case INT4, NF4 -> tensor.scales() != null;
            case INT8 -> tensor.scales() != null;
            case FP8 -> false;
        };
    }

    private void validateModelPath(Path modelPath) {
        if (!Files.exists(modelPath)) {
            throw new IllegalArgumentException("Model path does not exist: " + modelPath);
        }
        if (!Files.isDirectory(modelPath)) {
            throw new IllegalArgumentException("Model path is not a directory: " + modelPath);
        }
    }

    private Map<String, AccelTensor> loadWeights(Path modelPath) throws IOException {
        log.infof("Loading weights for quantization from %s", modelPath);
        try (SafetensorShardSession session = loader.open(modelPath)) {
            return weightBridge.bridgeAll(session);
        } catch (Exception e) {
            throw new IOException("Failed to load weights for quantization", e);
        }
    }

    private void saveWeights(Path outputPath, Map<String, AccelTensor> weights) throws IOException {
        log.infof("Saving quantized weights to %s", outputPath);
        AccelSafetensorWriter writer = new AccelSafetensorWriter();
        writer.save(outputPath, weights);
    }

    private long calculateTotalSize(Map<String, AccelTensor> weights) {
        return weights.values().stream()
                .mapToLong(this::estimateTensorSize)
                .sum();
    }

    private long estimateTensorSize(AccelTensor tensor) {
        // Estimate tensor size in bytes
        // Actual implementation would get dtype and shape from tensor
        return tensor != null ? 1024L : 0L; // Placeholder
    }

    private long countParameters(Map<String, AccelTensor> weights) {
        return weights.values().stream()
                .mapToLong(this::estimateParamCount)
                .sum();
    }

    private long estimateParamCount(AccelTensor tensor) {
        // Estimate parameter count
        // Actual implementation would get shape from tensor
        return tensor != null ? 256L : 0L; // Placeholder
    }

    private QuantConfig createDefaultConfig(QuantStrategy strategy) {
        return switch (strategy) {
            case INT4 -> QuantConfig.int4Gptq();
            case INT8 -> QuantConfig.int8();
            case FP8 -> QuantConfig.fp8();
            default -> new QuantConfig();
        };
    }

    /**
     * Shutdown the engine and release resources.
     */
    @PreDestroy
    public void shutdown() {
        ExecutorService current = executor;
        if (current != null) {
            current.shutdownNow();
            executor = null;
        }
        clearCache();
        log.info("Quantization engine shutdown complete");
    }
}
