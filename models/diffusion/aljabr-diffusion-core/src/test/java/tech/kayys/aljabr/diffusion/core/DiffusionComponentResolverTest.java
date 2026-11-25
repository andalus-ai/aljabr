package tech.kayys.aljabr.diffusion.core;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentManifest;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentRequirement;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionComponentResolverTest {

    private final DiffusionComponentResolver resolver = new DiffusionComponentResolver();

    @Test
    void resolvesRequiredAndOptionalRolesWithoutFamilyKnowledge() {
        var manifest = new DiffusionComponentManifest("org/model", "main", List.of(
                new DiffusionComponentArtifact("transformer", "diffusion-model",
                        "model/transformer.safetensors", "safetensors", true),
                new DiffusionComponentArtifact("encoder", "text-encoder",
                        "model/text_encoder.safetensors", "safetensors", true)));

        var result = resolver.resolve(manifest, List.of(
                new DiffusionComponentRequirement("diffusion-model", true),
                new DiffusionComponentRequirement("text-encoder", true),
                new DiffusionComponentRequirement("audio-autoencoder", false)));

        assertTrue(result.complete());
        assertEquals("transformer", result.require("diffusion-model").id());
        assertFalse(result.selected().containsKey("audio-autoencoder"));
    }

    @Test
    void reportsMissingRequiredRoles() {
        var result = resolver.resolve(
                new DiffusionComponentManifest("org/model", "main", List.of()),
                List.of(new DiffusionComponentRequirement("video-autoencoder", true)));

        assertFalse(result.complete());
        assertEquals(List.of("video-autoencoder"), result.missingRequiredRoles());
    }

    @Test
    void filtersByAcceptedFormatAndPrefersRequestedVariant() {
        var manifest = new DiffusionComponentManifest("org/model", "rev-1", List.of(
                new DiffusionComponentArtifact("fp16", "diffusion-model", "fp16.safetensors",
                        "safetensors", "fp16", null, true, java.util.Map.of()),
                new DiffusionComponentArtifact("bf16", "diffusion-model", "bf16.safetensors",
                        "safetensors", "bf16", null, true, java.util.Map.of()),
                new DiffusionComponentArtifact("other", "diffusion-model", "model.bin",
                        "pytorch", "bf16", null, true, java.util.Map.of())));

        var result = resolver.resolve(manifest, List.of(
                new DiffusionComponentRequirement("diffusion-model", true,
                        Set.of("safetensors"), "bf16")));

        assertEquals("bf16", result.require("diffusion-model").id());
    }

    @Test
    void keepsFirstCompatibleArtifactWhenNoPreferredVariantMatches() {
        var manifest = new DiffusionComponentManifest("org/model", "main", List.of(
                new DiffusionComponentArtifact("first", "text-encoder", "first.safetensors",
                        "safetensors", true),
                new DiffusionComponentArtifact("second", "text-encoder", "second.safetensors",
                        "safetensors", true)));

        var result = resolver.resolve(manifest, List.of(
                new DiffusionComponentRequirement("text-encoder", true, Set.of("safetensors"), "fp8")));

        assertEquals("first", result.require("text-encoder").id());
    }

    @Test
    void rejectsDuplicateArtifactIdsAndInvalidDigests() {
        assertThrows(IllegalArgumentException.class, () -> new DiffusionComponentManifest(
                "org/model", "main", List.of(
                        new DiffusionComponentArtifact("same", "text-encoder", "a", "safetensors", true),
                        new DiffusionComponentArtifact("same", "video-autoencoder", "b", "safetensors", true))));

        assertThrows(IllegalArgumentException.class, () -> new DiffusionComponentArtifact(
                "bad", "text-encoder", "a", "safetensors", null, "not-a-digest", true, java.util.Map.of()));
    }
}
