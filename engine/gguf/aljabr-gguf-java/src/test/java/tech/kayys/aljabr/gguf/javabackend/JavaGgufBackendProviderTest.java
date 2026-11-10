package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.api.GgufBackendProvider;

import java.util.ServiceLoader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the pure-Java GGUF backend provider registration and
 * properties. Tests that load a real model file are integration tests and live
 * in the test resources directory alongside actual GGUF fixtures.
 */
class JavaGgufBackendProviderTest {

    @Test
    void providerIsAlwaysAvailable() {
        JavaGgufBackendProvider provider = new JavaGgufBackendProvider();
        assertTrue(provider.isAvailable(), "Pure-Java backend should always be available");
    }

    @Test
    void providerIdIsJava() {
        JavaGgufBackendProvider provider = new JavaGgufBackendProvider();
        assertEquals("java", provider.id());
    }

    @Test
    void providerPriorityOutranksLlamaCpp() {
        JavaGgufBackendProvider provider = new JavaGgufBackendProvider();
        // Java (10) > llamacpp (-10): once Java becomes ready, AUTO must prefer it.
        assertTrue(provider.priority() > 0,
                "Java provider priority should be positive to outrank llama.cpp in AUTO mode");
    }

    @Test
    void providerAliasesIncludeExpectedTokens() {
        JavaGgufBackendProvider provider = new JavaGgufBackendProvider();
        assertTrue(provider.aliases().contains("java-native"));
        assertTrue(provider.aliases().contains("pure-java"));
    }

    @Test
    void serviceLoaderDiscoversJavaProvider() {
        boolean found = ServiceLoader.load(GgufBackendProvider.class).stream()
                .anyMatch(p -> p.type().equals(JavaGgufBackendProvider.class));
        assertTrue(found, "JavaGgufBackendProvider should be discoverable via ServiceLoader");
    }
}
