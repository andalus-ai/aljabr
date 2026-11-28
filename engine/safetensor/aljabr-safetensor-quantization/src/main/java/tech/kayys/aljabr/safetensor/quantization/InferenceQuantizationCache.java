/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;

/**
 * Owns the on-disk cache representation used by inference-time weight
 * quantization. Cache failures are deliberately non-fatal: callers can
 * continue by quantizing tensors without a cache.
 *
 * <p>The version-1 binary layout is retained for compatibility with existing
 * cache files.</p>
 */
public final class InferenceQuantizationCache {

    private static final System.Logger LOG = System.getLogger(InferenceQuantizationCache.class.getName());
    private static final int CACHE_VERSION = 1;

    /**
     * Resolves the cache file for a model and effective strategy.
     * Returns {@code null} when caching is disabled or the directory cannot
     * be prepared.
     */
    public Path cachePath(Path modelPath, QuantizationStrategy strategy) {
        if (modelPath == null || strategy == null || strategy == QuantizationStrategy.NONE) {
            return null;
        }
        try {
            String home = System.getProperty("aljabr.home",
                    Path.of(System.getProperty("user.home"), ".aljabr").toString());
            Path preferred = Path.of(home, "cache", "direct-quant");
            Path directory = ensureDirectory(preferred);
            Path normalized = modelPath.toAbsolutePath().normalize();
            long modified = Files.exists(normalized) ? Files.getLastModifiedTime(normalized).toMillis() : 0L;
            long size = Files.isRegularFile(normalized) ? Files.size(normalized) : 0L;
            String key = normalized + "|" + strategy.name() + "|" + modified + "|" + size;
            return directory.resolve(Integer.toHexString(key.hashCode()) + ".dqcache");
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "Failed to prepare inference quantization cache for " + modelPath, e);
            return null;
        }
    }

    private Path ensureDirectory(Path preferred) throws IOException {
        try {
            Files.createDirectories(preferred);
            return preferred;
        } catch (IOException preferredFailure) {
            Path fallback = Path.of(System.getProperty("java.io.tmpdir"), "aljabr-direct-quant");
            Files.createDirectories(fallback);
            LOG.log(System.Logger.Level.WARNING,
                    "Using temporary inference quantization cache directory: " + fallback, preferredFailure);
            return fallback;
        }
    }

    /**
     * Loads cached tensors. An absent, stale, malformed, or unreadable cache
     * is treated as a cache miss.
     */
    public Map<String, CachedTensor> load(Path cachePath, QuantizationStrategy strategy) {
        if (cachePath == null || strategy == null || !Files.isRegularFile(cachePath)) {
            return Collections.emptyMap();
        }
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(cachePath)))) {
            int version = in.readInt();
            if (version != CACHE_VERSION) {
                LOG.log(System.Logger.Level.WARNING,
                        "Ignoring inference cache with unsupported version " + version + " at " + cachePath);
                return Collections.emptyMap();
            }
            if (!strategy.name().equals(in.readUTF())) {
                return Collections.emptyMap();
            }
            int count = in.readInt();
            if (count < 0) {
                throw new IOException("Negative cached tensor count: " + count);
            }
            Map<String, CachedTensor> cached = new HashMap<>(Math.max(1, count * 2));
            for (int i = 0; i < count; i++) {
                String name = in.readUTF();
                int rank = in.readInt();
                if (rank < 0 || rank > 64) {
                    throw new IOException("Invalid cached tensor rank: " + rank);
                }
                long[] shape = new long[rank];
                for (int j = 0; j < rank; j++) {
                    shape[j] = in.readLong();
                }
                int quantOrdinal = in.readInt();
                AccelTensor.QuantType[] quantTypes = AccelTensor.QuantType.values();
                if (quantOrdinal < 0 || quantOrdinal >= quantTypes.length) {
                    throw new IOException("Invalid cached quantization type: " + quantOrdinal);
                }
                int groupSize = in.readInt();
                int dataLength = in.readInt();
                if (dataLength < 0) {
                    throw new IOException("Negative cached tensor data length: " + dataLength);
                }
                byte[] data = in.readNBytes(dataLength);
                if (data.length != dataLength) {
                    throw new IOException("Truncated cached tensor data for " + name);
                }
                int scalesLength = in.readInt();
                if (scalesLength < 0) {
                    throw new IOException("Negative cached scales length: " + scalesLength);
                }
                float[] scales = new float[scalesLength];
                for (int j = 0; j < scalesLength; j++) {
                    scales[j] = in.readFloat();
                }
                int zerosLength = in.readInt();
                if (zerosLength < 0) {
                    throw new IOException("Negative cached zeros length: " + zerosLength);
                }
                float[] zeros = new float[zerosLength];
                for (int j = 0; j < zerosLength; j++) {
                    zeros[j] = in.readFloat();
                }
                cached.put(name, new CachedTensor(shape, quantTypes[quantOrdinal],
                        groupSize, data, scales, zeros));
            }
            LOG.log(System.Logger.Level.DEBUG,
                    "Loaded inference quantization cache " + cachePath + " (" + count + " tensors)");
            return cached;
        } catch (Exception e) {
            LOG.log(System.Logger.Level.WARNING, "Failed to load inference quantization cache from " + cachePath, e);
            return Collections.emptyMap();
        }
    }

    /**
     * Persists eligible INT4 tensors using the existing version-1 cache layout.
     */
    public void persist(Path cachePath, QuantizationStrategy strategy, Map<String, AccelTensor> weights) {
        if (cachePath == null || strategy == null || weights == null) {
            return;
        }
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(cachePath)))) {
            List<Map.Entry<String, AccelTensor>> entries = weights.entrySet().stream()
                    .filter(entry -> isPersistable(entry.getValue()))
                    .sorted(Map.Entry.comparingByKey())
                    .toList();
            out.writeInt(CACHE_VERSION);
            out.writeUTF(strategy.name());
            out.writeInt(entries.size());
            for (Map.Entry<String, AccelTensor> entry : entries) {
                AccelTensor tensor = entry.getValue();
                out.writeUTF(entry.getKey());
                out.writeInt(tensor.rank());
                for (long dimension : tensor.shape()) {
                    out.writeLong(dimension);
                }
                out.writeInt(tensor.quantType().ordinal());
                out.writeInt(tensor.groupSize());
                byte[] data = tensor.dataSegment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                out.writeInt(data.length);
                out.write(data);
                float[] scales = tensor.scales() == null ? new float[0]
                        : tensor.scales().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
                out.writeInt(scales.length);
                for (float scale : scales) {
                    out.writeFloat(scale);
                }
                float[] zeros = tensor.zeros() == null ? new float[0]
                        : tensor.zeros().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
                out.writeInt(zeros.length);
                for (float zero : zeros) {
                    out.writeFloat(zero);
                }
            }
            LOG.log(System.Logger.Level.DEBUG,
                    "Saved inference quantization cache " + cachePath + " (" + entries.size() + " tensors)");
        } catch (Exception e) {
            LOG.log(System.Logger.Level.WARNING, "Failed to persist inference quantization cache to " + cachePath, e);
        }
    }

    private boolean isPersistable(AccelTensor tensor) {
        return tensor != null
                && tensor.quantType() == AccelTensor.QuantType.INT4
                && tensor.scales() != null;
    }

    /** Immutable cache payload; tensor materialization occurs only on demand. */
    public record CachedTensor(long[] shape, AccelTensor.QuantType quantType,
            int groupSize, byte[] data, float[] scales, float[] zeros) {

        public AccelTensor toTensor() {
            AccelTensor tensor = AccelTensor.fromByteArray(data, shape);
            java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
            java.lang.foreign.MemorySegment scaleSegment = scales.length == 0 ? null
                    : arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_FLOAT, scales);
            java.lang.foreign.MemorySegment zeroSegment = zeros.length == 0 ? null
                    : arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_FLOAT, zeros);
            return tensor.withQuantization(quantType, scaleSegment, zeroSegment, groupSize);
        }
    }
}
