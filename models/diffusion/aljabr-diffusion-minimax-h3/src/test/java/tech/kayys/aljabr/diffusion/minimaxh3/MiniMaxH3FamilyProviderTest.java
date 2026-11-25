package tech.kayys.aljabr.diffusion.minimaxh3;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionFamilyProvider;
import tech.kayys.aljabr.diffusion.api.DiffusionModelDescriptor;

import java.util.Map;
import java.util.ServiceLoader;

import static org.junit.jupiter.api.Assertions.*;

class MiniMaxH3FamilyProviderTest {

    private final MiniMaxH3FamilyProvider provider = new MiniMaxH3FamilyProvider();

    @Test
    void identifiesFamilyWithoutRequiringTaskMetadata() {
        assertTrue(provider.supports(new DiffusionModelDescriptor(
                MiniMaxH3FamilyProvider.FAMILY_ID,
                MiniMaxH3FamilyProvider.ARCHITECTURE_ID)));
    }

    @Test
    void recognizesDeclaredGenerationTasks() {
        for (MiniMaxH3Task task : MiniMaxH3Task.values()) {
            assertTrue(provider.supports(new DiffusionModelDescriptor(
                    MiniMaxH3FamilyProvider.FAMILY_ID,
                    MiniMaxH3FamilyProvider.ARCHITECTURE_ID,
                    Map.of("task", task.id()))), task.id());
        }
    }

    @Test
    void rejectsUnknownTaskAndOtherArchitectures() {
        assertFalse(provider.supports(new DiffusionModelDescriptor(
                MiniMaxH3FamilyProvider.FAMILY_ID,
                MiniMaxH3FamilyProvider.ARCHITECTURE_ID,
                Map.of("task", "speech-to-text"))));
        assertFalse(provider.supports(new DiffusionModelDescriptor(
                MiniMaxH3FamilyProvider.FAMILY_ID, "minimax-h3-llm")));
        assertFalse(provider.supports(new DiffusionModelDescriptor(
                "other-family", MiniMaxH3FamilyProvider.ARCHITECTURE_ID)));
    }

    @Test
    void providerIsDiscoverableThroughServiceLoader() {
        assertTrue(ServiceLoader.load(DiffusionFamilyProvider.class).stream()
                .map(ServiceLoader.Provider::get)
                .anyMatch(MiniMaxH3FamilyProvider.class::isInstance));
    }

    @Test
    void componentRolesCoverTheModelBundle() {
        assertArrayEquals(
                new MiniMaxH3ComponentRole[]{
                        MiniMaxH3ComponentRole.DIFFUSION_MODEL,
                        MiniMaxH3ComponentRole.TEXT_ENCODER,
                        MiniMaxH3ComponentRole.VIDEO_AUTOENCODER,
                        MiniMaxH3ComponentRole.AUDIO_AUTOENCODER,
                        MiniMaxH3ComponentRole.ADAPTER,
                        MiniMaxH3ComponentRole.MODEL_PATCH,
                        MiniMaxH3ComponentRole.PROMPT_EMBEDDING
                },
                MiniMaxH3ComponentRole.values());
    }
}
