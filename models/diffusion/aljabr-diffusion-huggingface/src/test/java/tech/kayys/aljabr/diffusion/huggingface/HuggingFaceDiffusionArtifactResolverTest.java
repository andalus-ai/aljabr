package tech.kayys.aljabr.diffusion.huggingface;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;

import static org.junit.jupiter.api.Assertions.*;

class HuggingFaceDiffusionArtifactResolverTest {
    private final HuggingFaceDiffusionArtifactResolver resolver =
            new HuggingFaceDiffusionArtifactResolver(java.net.http.HttpClient.newHttpClient(),
                    "https://huggingface.co", null);

    @Test
    void recognizesHuggingFaceArtifactUris() {
        assertTrue(resolver.supports(new DiffusionComponentArtifact(
                "weights", "diffusion-model",
                "hf://Comfy-Org/MiniMax-H3/model.safetensors?revision=main",
                "safetensors", true)));
    }

    @Test
    void rejectsOtherSchemesAndUnsafePaths() {
        assertFalse(resolver.supports(new DiffusionComponentArtifact(
                "weights", "diffusion-model", "s3://bucket/model.safetensors", "safetensors", true)));
        assertFalse(resolver.supports(new DiffusionComponentArtifact(
                "weights", "diffusion-model", "hf://owner/repo/../secret.safetensors", "safetensors", true)));
    }

    @Test
    void rejectsMissingRepositoryFile() {
        assertFalse(resolver.supports(new DiffusionComponentArtifact(
                "weights", "diffusion-model", "hf://owner/repo", "safetensors", true)));
    }
}
