/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizerRegistry.java
 * Central registry for built-in and optional quantizer implementations.
 */
package tech.kayys.aljabr.safetensor.quantization;

import tech.kayys.aljabr.safetensor.quantization.quantizer.FP8Quantizer;
import tech.kayys.aljabr.safetensor.quantization.quantizer.GPTQQuantizer;
import tech.kayys.aljabr.safetensor.quantization.quantizer.INT8Quantizer;
import tech.kayys.aljabr.safetensor.quantization.quantizer.Quantizer;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Registry of Aljabr built-in quantizers and optional providers discovered through
 * Java's {@link ServiceLoader}. Optional extensions can implement the Aljabr-owned
 * {@link Quantizer} SPI without introducing a dependency from Aljabr to the extension.
 */
public final class QuantizerRegistry {

    private static final Logger LOG = Logger.getLogger(QuantizerRegistry.class.getName());
    private static final Map<String, Quantizer> QUANTIZERS = new ConcurrentHashMap<>();

    static {
        register("GPTQQuantizer", new GPTQQuantizer());
        register("INT8Quantizer", new INT8Quantizer());
        register("FP8Quantizer", new FP8Quantizer());
        discoverProviders();
    }

    private QuantizerRegistry() {
    }

    private static void discoverProviders() {
        Iterator<Quantizer> providers = ServiceLoader.load(Quantizer.class).iterator();
        while (true) {
            final Quantizer provider;
            try {
                if (!providers.hasNext()) {
                    return;
                }
                provider = providers.next();
            } catch (ServiceConfigurationError error) {
                // ServiceLoader may repeatedly encounter a malformed descriptor.
                // Stop discovery rather than risking an infinite loop during startup.
                LOG.log(Level.WARNING, "Stopping optional quantizer discovery after a provider configuration error", error);
                return;
            }

            try {
                String name = normalize(provider.getName());
                Quantizer previous = QUANTIZERS.putIfAbsent(name, provider);
                if (previous != null) {
                    LOG.warning(() -> "Ignoring duplicate quantizer provider '" + name
                            + "' from " + provider.getClass().getName()
                            + "; already registered by " + previous.getClass().getName());
                } else {
                    LOG.info(() -> "Discovered optional quantizer provider: " + provider.getName());
                }
            } catch (RuntimeException error) {
                LOG.log(Level.WARNING, "Optional quantizer provider failed during registration", error);
            }
        }
    }

    private static String normalize(String name) {
        Objects.requireNonNull(name, "Quantizer name must not be null");
        String normalized = name.trim().toLowerCase(Locale.ROOT);
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("Quantizer name must not be blank");
        }
        return normalized;
    }

    /** Register or replace a quantizer by name. */
    public static void register(String name, Quantizer quantizer) {
        QUANTIZERS.put(normalize(name), Objects.requireNonNull(quantizer, "Quantizer instance must not be null"));
    }

    /** Return a registered quantizer, or null when it is unavailable. */
    public static Quantizer get(String name) {
        return QUANTIZERS.get(normalize(name));
    }

    /** Return an immutable snapshot of registered quantizers. */
    public static Map<String, Quantizer> getAll() {
        return Collections.unmodifiableMap(new TreeMap<>(QUANTIZERS));
    }

    public static Set<String> getNames() {
        return Collections.unmodifiableSet(new TreeSet<>(QUANTIZERS.keySet()));
    }

    public static boolean has(String name) {
        return QUANTIZERS.containsKey(normalize(name));
    }

    /** Select the first supporting quantizer in stable name order. */
    public static Quantizer selectBest(QuantConfig config) {
        Objects.requireNonNull(config, "Quantization config must not be null");
        for (Quantizer quantizer : getAll().values()) {
            if (quantizer.supports(config)) {
                return quantizer;
            }
        }
        return null;
    }

    /** Clear registry state. Intended for isolated tests and explicit reinitialization. */
    public static void clear() {
        QUANTIZERS.clear();
    }

    public static Map<String, String> getStats() {
        Map<String, String> stats = new LinkedHashMap<>();
        getAll().forEach((name, quantizer) -> stats.put(name,
                String.format("%s (supports=%s)", quantizer.getName(),
                        quantizer.supports(QuantConfig.int4Gptq()))));
        return stats;
    }
}
