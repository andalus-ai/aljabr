package tech.kayys.aljabr.spi.provider;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ProviderRegistryTest {
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
    }

    @Test
    void rejectsDuplicateProviderIds() {
        assertThrows(IllegalStateException.class, () -> ProviderRegistry.fromProviders(List.of(
                new TestProvider("vendor/duplicate"), new TestProvider("vendor/duplicate"))));
    }

    @Test
    void returnsEmptyForUnknownProviderAndHandlesEmptyRegistry() {
        ProviderRegistry registry = ProviderRegistry.fromProviders(List.of());
        assertTrue(registry.isEmpty());
        assertTrue(registry.find("vendor/missing").isEmpty());
    }

    private static final class TestProvider implements AljabrProvider {
        private final ProviderDescriptor descriptor;
        private boolean initialized;

        private TestProvider(String id) {
            descriptor = new ProviderDescriptor(id, "1.0.0", "1", Set.of());
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
