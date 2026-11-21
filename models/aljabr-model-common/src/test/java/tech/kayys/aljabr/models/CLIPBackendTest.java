package tech.kayys.aljabr.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

class CLIPBackendTest {

    @Test
    void promptEmbeddingsAreAllocatedOnTheConfiguredBackend() {
        CpuBackend backend = new CpuBackend();
        CLIP clip = new CLIP(backend);
        try (Tensor embeddings = clip.encode("backend-owned prompt")) {
            assertEquals(backend, clip.computeBackend());
            assertEquals(backend, embeddings.backend());
            assertEquals(1, embeddings.shape().dim(0));
            assertEquals(77, embeddings.shape().dim(1));
            assertEquals(768, embeddings.shape().dim(2));
        } finally {
            clip.close();
        }
    }
}
