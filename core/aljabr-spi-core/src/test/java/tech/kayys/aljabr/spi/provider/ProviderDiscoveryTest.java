package tech.kayys.aljabr.spi.provider;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ProviderDiscoveryTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void discoversWithExplicitClassLoaderWithoutInitializingProviders() throws Exception {
        HealthyProvider.initializeCalls = 0;
        try (URLClassLoader loader = loaderWith(HealthyProvider.class)) {
            ProviderDiscoveryResult result = ProviderDiscovery.discover(loader);

            assertEquals(1, result.providers().size());
            assertEquals("test/healthy", result.providers().getFirst().descriptor().id());
            assertTrue(result.failures().isEmpty());
            assertEquals(0, HealthyProvider.initializeCalls);
        }
    }

    @Test
    void isolatesProviderConstructionFailureAndContinuesDiscovery() throws Exception {
        try (URLClassLoader loader = loaderWith(ThrowingConstructorProvider.class, HealthyProvider.class)) {
            ProviderDiscoveryResult result = ProviderDiscovery.discover(loader);

            assertEquals(1, result.providers().size());
            assertEquals("test/healthy", result.providers().getFirst().descriptor().id());
            assertEquals(1, result.failures().size());
            assertTrue(result.failures().getFirst().message().contains("discovery failed"));
        }
    }

    @Test
    void excludesEveryProviderWhenIdentifiersConflict() throws Exception {
        try (URLClassLoader loader = loaderWith(DuplicateProviderA.class, DuplicateProviderB.class)) {
            ProviderDiscoveryResult result = ProviderDiscovery.discover(loader);

            assertTrue(result.providers().isEmpty());
            assertEquals(2, result.failures().size());
            assertTrue(result.failures().stream()
                    .allMatch(failure -> failure.message().contains("Duplicate provider id")));
        }
    }

    @Test
    void separateClassLoadersDiscoverIndependentProviderSets() throws Exception {
        try (URLClassLoader first = loaderWith(HealthyProvider.class);
             URLClassLoader second = loaderWith(OtherProvider.class)) {
            ProviderDiscoveryResult firstResult = ProviderDiscovery.discover(first);
            ProviderDiscoveryResult secondResult = ProviderDiscovery.discover(second);

            assertEquals("test/healthy", firstResult.providers().getFirst().descriptor().id());
            assertEquals("test/other", secondResult.providers().getFirst().descriptor().id());
            assertFalse(firstResult.providers().getFirst() == secondResult.providers().getFirst());
        }
    }

    private URLClassLoader loaderWith(Class<?>... providerTypes) throws IOException {
        Path services = temporaryDirectory.resolve("META-INF/services");
        Files.createDirectories(services);
        Path descriptor = services.resolve(AljabrProvider.class.getName());
        String content = String.join("\n", java.util.Arrays.stream(providerTypes)
                .map(Class::getName).toList()) + "\n";
        Files.writeString(descriptor, content);
        URL url = temporaryDirectory.toUri().toURL();
        return new URLClassLoader(new URL[]{url}, AljabrProvider.class.getClassLoader());
    }

    public static class HealthyProvider implements AljabrProvider {
        static int initializeCalls;

        public HealthyProvider() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptor("test/healthy");
        }

        @Override
        public void initialize(ProviderContext context) {
            initializeCalls++;
        }
    }

    public static class OtherProvider implements AljabrProvider {
        public OtherProvider() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptor("test/other");
        }
    }

    public static class ThrowingConstructorProvider implements AljabrProvider {
        public ThrowingConstructorProvider() {
            throw new IllegalStateException("intentional constructor failure");
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptor("test/broken");
        }
    }

    public static class DuplicateProviderA implements AljabrProvider {
        public DuplicateProviderA() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptor("test/duplicate");
        }
    }

    public static class DuplicateProviderB implements AljabrProvider {
        public DuplicateProviderB() {
        }

        @Override
        public ProviderDescriptor descriptor() {
            return descriptor("test/duplicate");
        }
    }

    private static ProviderDescriptor descriptor(String id) {
        return new ProviderDescriptor(id, "1.0.0", "1", java.util.Set.of());
    }
}
