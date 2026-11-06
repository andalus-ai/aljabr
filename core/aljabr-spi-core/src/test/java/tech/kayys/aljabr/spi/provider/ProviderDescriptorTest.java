package tech.kayys.aljabr.spi.provider;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
class ProviderDescriptorTest {
    @Test void acceptsNamespacedProviderAndCopiesCapabilities() {
        var capability = new CapabilityDescriptor("linear-algebra.dense", "1", CapabilityStability.STABLE);
        var capabilities = new java.util.HashSet<>(Set.of(capability));
        var descriptor = new ProviderDescriptor("andalus/cpu-reference", "0.1.0", ">=0.1.0 <1.0.0", capabilities);
        capabilities.clear();
        assertEquals("andalus/cpu-reference", descriptor.id());
        assertEquals(Set.of(capability), descriptor.capabilities());
    }
    @Test void rejectsUnqualifiedProviderId() {
        assertThrows(IllegalArgumentException.class,
                () -> new ProviderDescriptor("cpu", "0.1.0", "^0.1", Set.of()));
    }
    @Test void providerContextIsImmutableAndSupportsDefaults() {
        var source = new java.util.HashMap<>(Map.of("threads", "4"));
        var context = new ProviderContext("0.1.0", source);
        source.put("threads", "8");
        assertEquals("4", context.config("threads").orElseThrow());
        assertEquals("auto", context.config("device", "auto"));
        assertThrows(UnsupportedOperationException.class,
                () -> context.configuration().put("device", "gpu"));
    }
}
