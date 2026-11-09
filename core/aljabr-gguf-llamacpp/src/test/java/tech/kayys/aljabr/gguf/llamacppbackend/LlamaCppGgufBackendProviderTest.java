package tech.kayys.aljabr.gguf.llamacppbackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.model.ModelFormat;
import tech.kayys.aljabr.spi.model.ModelManifest;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class LlamaCppGgufBackendProviderTest {

    @Test
    void createsLocalGgufManifestForBackendInitialization() throws Exception {
        Path model = Files.createTempFile("provider-manifest", ".gguf");
        try {
            ModelManifest manifest = LlamaCppGgufBackendProvider.manifestFor(model);
            assertEquals(model.getFileName().toString(), manifest.modelId());
            assertTrue(manifest.supportsFormat(ModelFormat.GGUF));
            assertEquals(model.toAbsolutePath().normalize().toString(),
                    manifest.artifacts().get(ModelFormat.GGUF).uri());
            assertEquals("local", manifest.apiKey());
            assertNotNull(manifest.requestId());
        } finally {
            Files.deleteIfExists(model);
        }
    }
}
