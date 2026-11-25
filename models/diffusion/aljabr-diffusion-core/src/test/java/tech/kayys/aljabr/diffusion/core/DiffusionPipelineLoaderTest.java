package tech.kayys.aljabr.diffusion.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.diffusion.api.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionPipelineLoaderTest {
    @TempDir Path tempDir;

    @Test
    void resolvesMaterializesLoadsAndClosesComponentsInReverseOrder() throws Exception {
        Path first = Files.writeString(tempDir.resolve("transformer.bin"), "transformer");
        Path second = Files.writeString(tempDir.resolve("encoder.bin"), "encoder");
        List<String> events = new ArrayList<>();
        var loader = new DiffusionPipelineLoader(
                new DiffusionComponentResolver(),
                new DiffusionArtifactResolverRegistry(List.of(new LocalDiffusionArtifactResolver())),
                new DiffusionComponentLoaderRegistry(List.of(new RecordingLoader(events))));

        var manifest = new DiffusionComponentManifest("org/model", "rev-1", List.of(
                new DiffusionComponentArtifact("transformer", "diffusion-model", first.toString(), "test", true),
                new DiffusionComponentArtifact("encoder", "text-encoder", second.toString(), "test", true)));
        var requirements = List.of(
                new DiffusionComponentRequirement("diffusion-model", true),
                new DiffusionComponentRequirement("text-encoder", true));

        try (LoadedDiffusionPipeline pipeline = loader.load(manifest, requirements,
                new DiffusionArtifactResolutionContext(tempDir.resolve("cache"), Duration.ofSeconds(5), true))) {
            assertEquals("org/model", pipeline.modelId());
            assertEquals(Set.of("diffusion-model", "text-encoder"), pipeline.components().keySet());
            assertTrue(Files.isRegularFile(pipeline.artifacts().get("diffusion-model").localPath()));
        }

        assertEquals(List.of("load:transformer", "load:encoder", "close:encoder", "close:transformer"), events);
    }

    @Test
    void validatesAllRequiredRolesBeforeResolvingAnyArtifact() {
        List<String> events = new ArrayList<>();
        var loader = new DiffusionPipelineLoader(new DiffusionComponentResolver(),
                new DiffusionArtifactResolverRegistry(List.of(new LocalDiffusionArtifactResolver())),
                new DiffusionComponentLoaderRegistry(List.of(new RecordingLoader(events))));
        var manifest = new DiffusionComponentManifest("org/model", "main", List.of());

        var failure = assertThrows(DiffusionPipelineLoadException.class, () -> loader.load(
                manifest, List.of(new DiffusionComponentRequirement("text-encoder", true)),
                new DiffusionArtifactResolutionContext(tempDir.resolve("cache"), Duration.ofSeconds(5), true)));

        assertEquals(List.of("text-encoder"), failure.missingRequiredRoles());
        assertTrue(events.isEmpty());
    }

    @Test
    void closesPreviouslyLoadedComponentsWhenALaterLoaderFails() throws Exception {
        Path first = Files.writeString(tempDir.resolve("first.bin"), "first");
        Path second = Files.writeString(tempDir.resolve("second.bin"), "second");
        List<String> events = new ArrayList<>();
        DiffusionComponentLoader<AutoCloseable> failing = new DiffusionComponentLoader<>() {
            public String format() { return "fail"; }
            public AutoCloseable load(DiffusionComponentArtifact artifact) {
                throw new IllegalStateException("synthetic failure");
            }
        };
        var loader = new DiffusionPipelineLoader(new DiffusionComponentResolver(),
                new DiffusionArtifactResolverRegistry(List.of(new LocalDiffusionArtifactResolver())),
                new DiffusionComponentLoaderRegistry(List.of(new RecordingLoader(events), failing)));
        var manifest = new DiffusionComponentManifest("org/model", "main", List.of(
                new DiffusionComponentArtifact("one", "diffusion-model", first.toString(), "test", true),
                new DiffusionComponentArtifact("two", "text-encoder", second.toString(), "fail", true)));

        assertThrows(DiffusionPipelineLoadException.class, () -> loader.load(manifest,
                List.of(new DiffusionComponentRequirement("diffusion-model", true),
                        new DiffusionComponentRequirement("text-encoder", true)),
                new DiffusionArtifactResolutionContext(tempDir.resolve("cache"), Duration.ofSeconds(5), true)));
        assertEquals(List.of("load:one", "close:one"), events);
    }

    private static final class RecordingLoader implements DiffusionComponentLoader<AutoCloseable> {
        private final List<String> events;
        private RecordingLoader(List<String> events) { this.events = events; }
        public String format() { return "test"; }
        public AutoCloseable load(DiffusionComponentArtifact artifact) {
            events.add("load:" + artifact.id());
            return () -> events.add("close:" + artifact.id());
        }
    }
}
