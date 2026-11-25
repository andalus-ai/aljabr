package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentLoadException;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentLoader;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** Deterministic format-to-loader registry for diffusion components. */
public final class DiffusionComponentLoaderRegistry {
    private final Map<String, DiffusionComponentLoader<? extends AutoCloseable>> loaders;

    public DiffusionComponentLoaderRegistry(
            Collection<? extends DiffusionComponentLoader<? extends AutoCloseable>> loaders) {
        Objects.requireNonNull(loaders, "loaders");
        Map<String, DiffusionComponentLoader<? extends AutoCloseable>> indexed = new LinkedHashMap<>();
        for (DiffusionComponentLoader<? extends AutoCloseable> loader : loaders) {
            Objects.requireNonNull(loader, "loader");
            String format = normalize(loader.format());
            if (indexed.putIfAbsent(format, loader) != null) {
                throw new IllegalArgumentException("Duplicate diffusion component loader for format: " + format);
            }
        }
        this.loaders = Map.copyOf(indexed);
    }

    public boolean supports(DiffusionComponentArtifact artifact) {
        if (artifact == null) return false;
        DiffusionComponentLoader<? extends AutoCloseable> loader = loaders.get(normalize(artifact.format()));
        return loader != null && loader.supports(artifact);
    }

    AutoCloseable load(DiffusionComponentArtifact artifact) {
        DiffusionComponentLoader<? extends AutoCloseable> loader =
                loaders.get(normalize(artifact.format()));
        if (loader == null || !loader.supports(artifact)) {
            throw new DiffusionComponentLoadException(
                    artifact, "No diffusion component loader supports format: " + artifact.format());
        }
        return loadUnchecked(loader, artifact);
    }

    private static AutoCloseable loadUnchecked(
            DiffusionComponentLoader<? extends AutoCloseable> loader,
            DiffusionComponentArtifact artifact) {
        return loader.load(artifact);
    }

    private static String normalize(String value) {
        Objects.requireNonNull(value, "loader format");
        String normalized = value.toLowerCase(Locale.ROOT);
        if (!normalized.matches("[a-z][a-z0-9+._-]*")) {
            throw new IllegalArgumentException("Invalid diffusion component format: " + value);
        }
        return normalized;
    }
}
