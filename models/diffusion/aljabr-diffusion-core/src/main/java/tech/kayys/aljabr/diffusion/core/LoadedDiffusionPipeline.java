package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.ResolvedDiffusionArtifact;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Caller-owned collection of loaded component handles. Closing the pipeline
 * closes every component in reverse load order, preserving suppressed errors.
 */
public final class LoadedDiffusionPipeline implements AutoCloseable {
    private final String modelId;
    private final String revision;
    private final Map<String, ResolvedDiffusionArtifact> artifacts;
    private final Map<String, AutoCloseable> components;
    private boolean closed;

    LoadedDiffusionPipeline(
            String modelId,
            String revision,
            Map<String, ResolvedDiffusionArtifact> artifacts,
            Map<String, AutoCloseable> components) {
        this.modelId = Objects.requireNonNull(modelId, "modelId");
        this.revision = Objects.requireNonNull(revision, "revision");
        this.artifacts = Collections.unmodifiableMap(new LinkedHashMap<>(artifacts));
        this.components = Collections.unmodifiableMap(new LinkedHashMap<>(components));
    }

    public String modelId() { return modelId; }
    public String revision() { return revision; }
    public Map<String, ResolvedDiffusionArtifact> artifacts() { return artifacts; }
    public Map<String, AutoCloseable> components() { return components; }
    public boolean isClosed() { return closed; }

    public AutoCloseable require(String role) {
        if (closed) throw new IllegalStateException("Diffusion pipeline is closed");
        AutoCloseable component = components.get(role);
        if (component == null) throw new IllegalStateException("No loaded diffusion component for role: " + role);
        return component;
    }

    @Override
    public void close() throws Exception {
        if (closed) return;
        closed = true;
        List<AutoCloseable> reverse = new ArrayList<>(components.values());
        Collections.reverse(reverse);
        Exception failure = null;
        for (AutoCloseable component : reverse) {
            try {
                component.close();
            } catch (Exception e) {
                if (failure == null) failure = e;
                else failure.addSuppressed(e);
            }
        }
        if (failure != null) throw failure;
    }
}
