package tech.kayys.aljabr.diffusion.minimaxh3;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentManifest;
import tech.kayys.aljabr.diffusion.core.DiffusionComponentResolver;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class MiniMaxH3PipelineRequirementsTest {

    @Test
    void textToVideoRequirementsResolveThroughGenericComponentResolver() {
        var requirements = MiniMaxH3PipelineRequirements.forTask(
                MiniMaxH3Task.TEXT_TO_VIDEO,
                MiniMaxH3PipelineRequirements.Conditioning.LOCAL_TEXT_ENCODER,
                MiniMaxH3PipelineRequirements.Output.VIDEO);
        var manifest = new DiffusionComponentManifest("Comfy-Org/MiniMax-H3", "main", List.of(
                artifact("transformer", MiniMaxH3ComponentRole.DIFFUSION_MODEL),
                artifact("text-encoder", MiniMaxH3ComponentRole.TEXT_ENCODER),
                artifact("video-vae", MiniMaxH3ComponentRole.VIDEO_AUTOENCODER)));

        var resolution = new DiffusionComponentResolver().resolve(manifest, requirements);

        assertTrue(resolution.complete());
        assertEquals(Set.of("diffusion-model", "text-encoder", "video-autoencoder"),
                resolution.selected().keySet());
    }

    @Test
    void precomputedConditioningDoesNotRequireLocalTextEncoder() {
        var requirements = MiniMaxH3PipelineRequirements.forTask(
                MiniMaxH3Task.TEXT_TO_IMAGE,
                MiniMaxH3PipelineRequirements.Conditioning.PRECOMPUTED_PROMPT_EMBEDDING,
                MiniMaxH3PipelineRequirements.Output.IMAGE);

        assertEquals(List.of("diffusion-model", "prompt-embedding"),
                requirements.stream().map(r -> r.role()).toList());
    }

    @Test
    void audioDecoderIsAddedOnlyWhenAudioOutputIsRequested() {
        var base = MiniMaxH3PipelineRequirements.forTask(
                MiniMaxH3Task.TEXT_TO_VIDEO,
                MiniMaxH3PipelineRequirements.Conditioning.LOCAL_TEXT_ENCODER,
                MiniMaxH3PipelineRequirements.Output.VIDEO);

        var withAudio = MiniMaxH3PipelineRequirements.withAudioOutput(base);

        assertFalse(base.stream().anyMatch(r -> r.role().equals("audio-autoencoder")));
        assertTrue(withAudio.stream().anyMatch(r ->
                r.role().equals("audio-autoencoder") && r.required()));
    }

    @Test
    void rejectsOutputModalityThatDoesNotMatchTask() {
        assertThrows(IllegalArgumentException.class, () ->
                MiniMaxH3PipelineRequirements.forTask(
                        MiniMaxH3Task.IMAGE_TO_VIDEO,
                        MiniMaxH3PipelineRequirements.Conditioning.LOCAL_TEXT_ENCODER,
                        MiniMaxH3PipelineRequirements.Output.IMAGE));
    }

    private static DiffusionComponentArtifact artifact(String id, MiniMaxH3ComponentRole role) {
        return new DiffusionComponentArtifact(
                id, role.roleId(), id + ".safetensors", "safetensors", null, null, true, Map.of());
    }
}
