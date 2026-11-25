package tech.kayys.aljabr.diffusion.core;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentLoader;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionComponentLoaderTest {
    @Test
    void defaultSupportMatchesArtifactFormat() {
        DiffusionComponentLoader<Loaded> loader = new TestLoader();
        assertTrue(loader.supports(artifact("safetensors")));
        assertFalse(loader.supports(artifact("gguf")));
        assertFalse(loader.supports(null));
    }

    @Test
    void returnedHandleIsCallerOwned() throws Exception {
        Loaded loaded = new TestLoader().load(artifact("safetensors"));
        assertFalse(loaded.closed);
        loaded.close();
        assertTrue(loaded.closed);
    }

    private static DiffusionComponentArtifact artifact(String format) {
        return new DiffusionComponentArtifact("encoder", "text-encoder",
                "file:///models/encoder.safetensors", format, true);
    }

    private static final class TestLoader implements DiffusionComponentLoader<Loaded> {
        public String format() { return "safetensors"; }
        public Loaded load(DiffusionComponentArtifact artifact) {
            if (!supports(artifact)) throw new IllegalArgumentException("Unsupported format");
            return new Loaded();
        }
    }

    private static final class Loaded implements AutoCloseable {
        private boolean closed;
        public void close() { closed = true; }
    }
}
