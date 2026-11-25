package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionContext;
import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionException;
import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolver;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.ResolvedDiffusionArtifact;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** Deterministic resolver selection by URI scheme. */
public final class DiffusionArtifactResolverRegistry {
    private final Map<String, DiffusionArtifactResolver> resolvers;

    public DiffusionArtifactResolverRegistry(Collection<? extends DiffusionArtifactResolver> resolvers) {
        Objects.requireNonNull(resolvers, "resolvers");
        Map<String, DiffusionArtifactResolver> indexed = new LinkedHashMap<>();
        for (DiffusionArtifactResolver resolver : resolvers) {
            Objects.requireNonNull(resolver, "resolver");
            String scheme = normalizeScheme(resolver.scheme());
            if (indexed.putIfAbsent(scheme, resolver) != null) {
                throw new IllegalArgumentException("Duplicate diffusion artifact resolver for scheme: " + scheme);
            }
        }
        this.resolvers = Map.copyOf(indexed);
    }

    public ResolvedDiffusionArtifact resolve(
            DiffusionComponentArtifact artifact, DiffusionArtifactResolutionContext context) {
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(context, "context");
        String scheme = schemeOf(artifact.location());
        DiffusionArtifactResolver resolver = resolvers.get(scheme);
        if (resolver == null || !resolver.supports(artifact)) {
            throw new DiffusionArtifactResolutionException(
                    artifact, "No diffusion artifact resolver supports location scheme: " + scheme);
        }
        return resolver.resolve(artifact, context);
    }

    public boolean supports(DiffusionComponentArtifact artifact) {
        if (artifact == null) return false;
        DiffusionArtifactResolver resolver = resolvers.get(schemeOf(artifact.location()));
        return resolver != null && resolver.supports(artifact);
    }

    private static String schemeOf(String location) {
        int separator = location.indexOf(':');
        return separator <= 0 ? "file" : normalizeScheme(location.substring(0, separator));
    }

    private static String normalizeScheme(String scheme) {
        Objects.requireNonNull(scheme, "resolver scheme");
        String normalized = scheme.toLowerCase(Locale.ROOT);
        if (!normalized.matches("[a-z][a-z0-9+.-]*")) {
            throw new IllegalArgumentException("Invalid resolver URI scheme: " + scheme);
        }
        return normalized;
    }
}
