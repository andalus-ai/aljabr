package tech.kayys.aljabr.diffusion.minimaxh3;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentManifest;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentRequirement;
import tech.kayys.aljabr.diffusion.core.DiffusionComponentResolver;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class MiniMaxH3ComponentManifestTest {

    @Test
    void minimaxRolesResolveThroughFamilyNeutralManifestContracts() {
        var manifest = new DiffusionComponentManifest("Comfy-Org/MiniMax-H3", "main", List.of(
                new DiffusionComponentArtifact("transformer", MiniMaxH3ComponentRole.DIFFUSION_MODEL.roleId(),
                        "diffusion_model.safetensors", "safetensors", "bf16", null, true, Map.of()),
                new DiffusionComponentArtifact("encoder", MiniMaxH3ComponentRole.TEXT_ENCODER.roleId(),
                        "text_encoder/model.safetensors", "safetensors", null, null, true, Map.of()),
                new DiffusionComponentArtifact("vae", MiniMaxH3ComponentRole.VIDEO_AUTOENCODER.roleId(),
                        "vae/video.safetensors", "safetensors", null, null, true, Map.of())));

        var result = new DiffusionComponentResolver().resolve(manifest, List.of(
                new DiffusionComponentRequirement(
                        MiniMaxH3ComponentRole.DIFFUSION_MODEL.roleId(), true, Set.of("safetensors"), "bf16"),
                new DiffusionComponentRequirement(
                        MiniMaxH3ComponentRole.TEXT_ENCODER.roleId(), true),
                new DiffusionComponentRequirement(
                        MiniMaxH3ComponentRole.VIDEO_AUTOENCODER.roleId(), true),
                new DiffusionComponentRequirement(
                        MiniMaxH3ComponentRole.AUDIO_AUTOENCODER.roleId(), false)));

        assertTrue(result.complete());
        assertEquals("transformer", result.require("diffusion-model").id());
        assertEquals("encoder", result.require("text-encoder").id());
        assertEquals("vae", result.require("video-autoencoder").id());
        assertFalse(result.selected().containsKey("audio-autoencoder"));
    }
}
