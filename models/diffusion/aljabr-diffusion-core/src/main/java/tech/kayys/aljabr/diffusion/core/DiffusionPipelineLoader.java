package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionContext;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentManifest;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentRequirement;
import tech.kayys.aljabr.diffusion.api.ResolvedDiffusionArtifact;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Coordinates capability selection, repository materialization, and
 * format-specific loading without embedding any model-family implementation.
 */
public final class DiffusionPipelineLoader {
    private final DiffusionComponentResolver componentResolver;
    private final DiffusionArtifactResolverRegistry artifactResolvers;
    private final DiffusionComponentLoaderRegistry componentLoaders;

    public DiffusionPipelineLoader(
            DiffusionComponentResolver componentResolver,
            DiffusionArtifactResolverRegistry artifactResolvers,
            DiffusionComponentLoaderRegistry componentLoaders) {
        this.componentResolver = Objects.requireNonNull(componentResolver, "componentResolver");
        this.artifactResolvers = Objects.requireNonNull(artifactResolvers, "artifactResolvers");
        this.componentLoaders = Objects.requireNonNull(componentLoaders, "componentLoaders");
    }

    /**
     * Resolve all required roles before doing I/O, then materialize and load
     * each selected component. Any failure closes resources already opened.
     */
    public LoadedDiffusionPipeline load(
            DiffusionComponentManifest manifest,
            List<DiffusionComponentRequirement> requirements,
            DiffusionArtifactResolutionContext context) {
        Objects.requireNonNull(manifest, "manifest");
        Objects.requireNonNull(requirements, "requirements");
        Objects.requireNonNull(context, "context");

        DiffusionComponentResolution selection = componentResolver.resolve(manifest, requirements);
        if (!selection.complete()) {
            throw new DiffusionPipelineLoadException(
                    "Diffusion manifest is missing required component roles: "
                            + selection.missingRequiredRoles(),
                    selection.missingRequiredRoles());
        }

        Map<String, ResolvedDiffusionArtifact> resolved = new LinkedHashMap<>();
        Map<String, AutoCloseable> loaded = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, DiffusionComponentArtifact> entry : selection.selected().entrySet()) {
                String role = entry.getKey();
                ResolvedDiffusionArtifact artifact = artifactResolvers.resolve(entry.getValue(), context);
                if (!componentLoaders.supports(artifact.asLocalArtifact())) {
                    throw new DiffusionPipelineLoadException(
                            "No format loader is registered for selected component role '"
                                    + role + "' with format '" + artifact.source().format() + "'",
                            List.of());
                }
                resolved.put(role, artifact);
                loaded.put(role, componentLoaders.load(artifact.asLocalArtifact()));
            }
            return new LoadedDiffusionPipeline(manifest.modelId(), manifest.revision(), resolved, loaded);
        } catch (Throwable failure) {
            closeAfterFailure(loaded, failure);
            if (failure instanceof Error error) throw error;
            if (failure instanceof DiffusionPipelineLoadException pipelineFailure) throw pipelineFailure;
            throw new DiffusionPipelineLoadException("Failed to load diffusion component pipeline", failure);
        }
    }

    private static void closeAfterFailure(Map<String, AutoCloseable> loaded, Throwable failure) {
        List<AutoCloseable> reverse = new ArrayList<>(loaded.values());
        java.util.Collections.reverse(reverse);
        for (AutoCloseable component : reverse) {
            try {
                component.close();
            } catch (Throwable closeFailure) {
                failure.addSuppressed(closeFailure);
            }
        }
    }
}
