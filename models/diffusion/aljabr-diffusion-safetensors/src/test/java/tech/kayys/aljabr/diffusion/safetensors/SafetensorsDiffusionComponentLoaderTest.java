package tech.kayys.aljabr.diffusion.safetensors;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class SafetensorsDiffusionComponentLoaderTest {

    @Test
    void resolvesPlainLocalPath() {
        assertEquals(Path.of("weights/model.safetensors").toAbsolutePath().normalize(),
                SafetensorsDiffusionComponentLoader.resolveLocalPath("weights/model.safetensors"));
    }

    @Test
    void resolvesFileUri() {
        Path expected = Path.of("/tmp/model.safetensors").toAbsolutePath().normalize();
        assertEquals(expected,
                SafetensorsDiffusionComponentLoader.resolveLocalPath("file:///tmp/model.safetensors"));
    }

    @Test
    void rejectsRemoteSchemesRatherThanPretendingToDownload() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> SafetensorsDiffusionComponentLoader.resolveLocalPath(
                        "hf://org/model/weights.safetensors"));
        assertTrue(error.getMessage().contains("local path or file URI"));
    }

    @Test
    void rejectsBlankLocations() {
        assertThrows(IllegalArgumentException.class,
                () -> SafetensorsDiffusionComponentLoader.resolveLocalPath(" "));
    }
}
