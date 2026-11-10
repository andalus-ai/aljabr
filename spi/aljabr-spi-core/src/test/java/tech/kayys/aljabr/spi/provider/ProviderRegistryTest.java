package tech.kayys.aljabr.spi.provider;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ProviderRegistryTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void indexesProvidersByNamespacedIdWithoutInitializingThem() {
        var first = new TestProvider("vendor/first");
        var second = new TestProvider("vendor/second");

        ProviderRegistry registry = ProviderRegistry.fromProviders(List.of(second, first));

        assertEquals(2, registry.size());
        assertSame(first, registry.find("vendor/first").orElseThrow());
        assertSame(second, registry.find("vendor/second").orElseThrow());
        assertFalse(first.initialized);
        assertFalse(second.initialized);
        assertThrows(UnsupportedOperationException.class, () -> registry.providers().clear());
        assertTrue(registry.discoveryFailures().isEmpty());
    }

    @Test
    void rejectsDuplicateProviderIdsInDirectRegistryConstruction() {
        assertThrows(IllegalStateException.class, () -> ProviderRegistry.fromProviders(List.of(
                new TestProvider("vendor/duplicate"), new TestProvider("vendor/duplicate"))));
    }

    @Test
    void discoveryKeepsHealthyProvidersWhenAnotherProviderCannotBeConstructed() throws Exception {
        try (URLClassLoader loader = loaderWith(BrokenProvider.class, HealthyProvider.class)) {
            ProviderRegistry registry = ProviderRegistry.discover(loader);

            assertEquals(1, registry.size());
            assertTrue(registry.find("test/healthy").isPresent());
            assertEquals(1, registry.discoveryFailures().size());
            assertTrue(registry.discoveryFailures().getFirst().message().contains("discovery failed"));
        }
    }

    @Test
    void discoveryExcludesEveryProviderWithADuplicateIdAndRetainsDiagnostics() throws Exception {
        try (URLClassLoader loader = loaderWith(DuplicateProviderA.class, DuplicateProviderB.class)) {
            ProviderRegistry registry = ProviderRegistry.discover(loader);

            assertTrue(registry.isEmpty());
            assertEquals(2, registry.discoveryFailures().size());
            assertTrue(registry.discoveryFailures().stream()
                    .allMatch(failure -> failure.message().contains("Duplicate provider id")));
        }
    }

    @Test
    void returnsEmptyForUnknownProviderAndHandlesEmptyRegistry() {
        ProviderRegistry registry = ProviderRegistry.fromProviders(List.of());
        assertTrue(registry.isEmpty());
        assertTrue(registry.find("vendor/missing").isEmpty());
        assertTrue(registry.discoveryFailures().isEmpty());
    }

    private URLClassLoader loaderWith(Class<?>... providerTypes) throws IOException {
        Path loaderDirectory = Files.createTempDirectory(temporaryDirectory, "provider-loader-");
        Path services = loaderDirectory.resolve("META-INF/services");
        Files.createDirectories(services);
        Path descriptor = services.resolve(AljabrProvider.class.getName());
        String content = String.join("\n", java.util.Arrays.stream(providerTypes)
                .map(Class::getName).toList()) + "\n";
        Files.writeString(descriptor, content);
        URL url = loaderDirectory.toUri().toURL();
        return new URLClassLoader(new URL[]{url}, AljabrProvider.class.getClassLoader());
    }

    public static class HealthyProvider implements AljabrProvider {
        public HealthyProvider() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptorFor("test/healthy");
        }
    }

    public static class BrokenProvider implements AljabrProvider {
        public BrokenProvider() {
            throw new IllegalStateException("intentional constructor failure");
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptorFor("test/broken");
        }
    }

    public static class DuplicateProviderA implements AljabrProvider {
        public DuplicateProviderA() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptorFor("test/duplicate");
        }
    }

    public static class DuplicateProviderB implements AljabrProvider {
        public DuplicateProviderB() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptorFor("test/duplicate");
        }
    }

    private static ProviderDescriptor descriptorFor(String id) {
        return new ProviderDescriptor(id, "1.0.0", ">=1.0.0 <2.0.0", Set.of());
    }

    private static final class TestProvider implements AljabrProvider {
        private final ProviderDescriptor descriptor;
        private boolean initialized;

        private TestProvider(String id) {
            descriptor = new ProviderDescriptor(id, "1.0.0", ">=1.0.0 <2.0.0", Set.of());
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptor;
        }

        @Override
        public void initialize(ProviderContext context) {
            initialized = true;
        }
    }
}
