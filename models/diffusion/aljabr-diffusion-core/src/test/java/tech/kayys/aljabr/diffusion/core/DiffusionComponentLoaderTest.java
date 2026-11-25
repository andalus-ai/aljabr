package tech.kayys.aljabr.diffusion.core;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentLoader;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionComponentLoaderTest {

    @Test
    void defaultSupportMatchesArtifactFormat() {
        DiffusionComponentLoader<LoadedComponent> loader = new TestLoader();
        assertTrue(loader.supports(artifact("safetensors")));
        assertFalse(loader.supports(artifact("gguf")));
        assertFalse(loader.supports(null));
    }

    @Test
    void loaderReturnsCallerOwnedCloseableHandle() {
        TestLoader loader = new TestLoader();
        LoadedComponent loaded = loader.load(artifact("safetensors"));
        assertEquals("weights.safetensors", loaded.location());
        assertFalse(loaded.closed());
        loaded.close();
        assertTrue(loaded.closed());
    }

    private static DiffusionComponentArtifact artifact(String format) {
        return new DiffusionComponentArtifact("text-encoder", "text-encoder",
                "file:///models/text/weights.safetensors", format, true);
    }

    private static final class TestLoader implements DiffusionComponentLoader<LoadedComponent> {
        @Override public String format() { return "safetensors"; }
        @Override public LoadedComponent load(DiffusionComponentArtifact artifact) {
            if (!supports(artifact)) throw new IllegalArgumentException("Unsupported format: " + artifact.format());
            return new LoadedComponent("weights.safetensors");
        }
    }

    private static final class LoadedComponent implements AutoCloseable {
        private final String location;
        private boolean closed;
        private LoadedComponent(String location) { this.location = location; }
        String location() { return location; }
        boolean closed() { return closed; }
        @Override public void close() { closed = true; }
    }
}
