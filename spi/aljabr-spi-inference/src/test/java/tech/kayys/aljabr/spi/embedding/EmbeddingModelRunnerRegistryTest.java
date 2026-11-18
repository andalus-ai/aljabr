package tech.kayys.aljabr.spi.embedding;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingModelRunnerRegistryTest {
    private static final String SERVICE_FILE =
            "META-INF/services/" + EmbeddingModelRunner.class.getName();

    @TempDir
    Path tempDir;

    @Test
    void discoversRunnerByModelFamilyWithoutInitializingModelRuntime() throws Exception {
        ClassLoader loader = serviceLoader(FirstRunner.class);

        EmbeddingModelRunnerRegistry registry = EmbeddingModelRunnerRegistry.discover(loader);

        assertEquals(1, registry.size());
        assertTrue(registry.find("embedding_gemma2").isPresent());
        assertEquals("embedding_gemma2", registry.find("embedding_gemma2").orElseThrow().modelFamily());
        assertTrue(registry.discoveryFailures().isEmpty());
    }

    @Test
    void excludesAllRunnersWhenModelFamilyIsClaimedMoreThanOnce() throws Exception {
        ClassLoader loader = serviceLoader(FirstRunner.class, DuplicateRunner.class);

        EmbeddingModelRunnerRegistry registry = EmbeddingModelRunnerRegistry.discover(loader);

        assertTrue(registry.isEmpty());
        assertFalse(registry.find("embedding_gemma2").isPresent());
        assertEquals(2, registry.discoveryFailures().size());
    }

    @Test
    void isolatesProviderConstructionFailure() throws Exception {
        ClassLoader loader = serviceLoader(FirstRunner.class, FailingRunner.class);

        EmbeddingModelRunnerRegistry registry = EmbeddingModelRunnerRegistry.discover(loader);

        assertEquals(1, registry.size());
        assertTrue(registry.find("embedding_gemma2").isPresent());
        assertEquals(1, registry.discoveryFailures().size());
        assertTrue(registry.discoveryFailures().getFirst().message()
                .contains("Embedding runner discovery failed"));
    }

    private ClassLoader serviceLoader(Class<?>... providers) throws Exception {
        Path servicePath = tempDir.resolve(SERVICE_FILE);
        Files.createDirectories(servicePath.getParent());
        String declarations = java.util.Arrays.stream(providers)
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining("\n"));
        Files.writeString(servicePath, declarations + "\n");
        URL[] urls = {tempDir.toUri().toURL()};
        return new URLClassLoader(urls, getClass().getClassLoader());
    }

    public static class FirstRunner implements EmbeddingModelRunner {
        @Override
        public String modelFamily() {
            return "embedding_gemma2";
        }

        @Override
        public java.util.concurrent.CompletionStage<EmbeddingResponse> embed(EmbeddingRequest request) {
            return CompletableFuture.failedFuture(new UnsupportedOperationException("test fixture"));
        }
    }

    public static class DuplicateRunner implements EmbeddingModelRunner {
        @Override
        public String modelFamily() {
            return "embedding_gemma2";
        }

        @Override
        public java.util.concurrent.CompletionStage<EmbeddingResponse> embed(EmbeddingRequest request) {
            return CompletableFuture.failedFuture(new UnsupportedOperationException("test fixture"));
        }
    }

    public static class FailingRunner implements EmbeddingModelRunner {
        public FailingRunner() {
            throw new IllegalStateException("fixture construction failure");
        }

        @Override
        public String modelFamily() {
            return "failing";
        }

        @Override
        public java.util.concurrent.CompletionStage<EmbeddingResponse> embed(EmbeddingRequest request) {
            return CompletableFuture.failedFuture(new UnsupportedOperationException("test fixture"));
        }
    }
}
