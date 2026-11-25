package tech.kayys.aljabr.diffusion.core;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.*;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionArtifactResolverRegistryTest {
    @Test
    void selectsResolverBySchemeAndReturnsResolvedArtifact() {
        DiffusionArtifactResolver resolver = new TestResolver("hf");
        DiffusionArtifactResolverRegistry registry =
                new DiffusionArtifactResolverRegistry(List.of(resolver));
        DiffusionComponentArtifact artifact = artifact("hf://org/model/encoder.safetensors");

        ResolvedDiffusionArtifact resolved = registry.resolve(
                artifact, new DiffusionArtifactResolutionContext(Path.of("cache"), java.time.Duration.ofSeconds(20), true));

        assertEquals(Path.of("cache/encoder.safetensors").toAbsolutePath().normalize(), resolved.localPath());
        assertEquals("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                resolved.sha256());
        assertEquals("file:", resolved.asLocalArtifact().location().substring(0, 5));
    }

    @Test
    void rejectsUnknownSchemes() {
        DiffusionArtifactResolverRegistry registry = new DiffusionArtifactResolverRegistry(List.of());
        assertThrows(DiffusionArtifactResolutionException.class,
                () -> registry.resolve(artifact("s3://bucket/model.safetensors"),
                        new DiffusionArtifactResolutionContext(Path.of("cache"),
                                java.time.Duration.ofSeconds(10), true)));
    }

    @Test
    void rejectsDuplicateSchemes() {
        assertThrows(IllegalArgumentException.class, () ->
                new DiffusionArtifactResolverRegistry(List.of(
                        new TestResolver("hf"), new TestResolver("HF"))));
    }

    private static DiffusionComponentArtifact artifact(String location) {
        return new DiffusionComponentArtifact("encoder", "text-encoder", location,
                "safetensors", null, null, true, Map.of());
    }

    private static final class TestResolver implements DiffusionArtifactResolver {
        private final String scheme;
        private TestResolver(String scheme) { this.scheme = scheme; }
        public String scheme() { return scheme; }
        public boolean supports(DiffusionComponentArtifact artifact) {
            return artifact.location().startsWith("hf://");
        }
        public ResolvedDiffusionArtifact resolve(
                DiffusionComponentArtifact artifact, DiffusionArtifactResolutionContext context) {
            return new ResolvedDiffusionArtifact(artifact, context.cacheDirectory().resolve("encoder.safetensors"),
                    "immutable-commit-sha", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", false);
        }
    }
}
