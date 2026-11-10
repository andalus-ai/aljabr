package tech.kayys.aljabr.spi.testkit;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.CapabilityDescriptor;
import tech.kayys.aljabr.spi.provider.CapabilityStability;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDescriptor;
import tech.kayys.aljabr.spi.provider.ProviderException;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class ProviderConformanceTest {
    private static final ProviderContext CONTEXT =
            new ProviderContext("0.1.0-test", Map.of());

    @Test
    void acceptsProviderWithValidDescriptorAndLifecycle() {
        AtomicBoolean initialized = new AtomicBoolean();
        AtomicBoolean closed = new AtomicBoolean();
        AljabrProvider provider = provider(() -> initialized.set(true), () -> closed.set(true));

        ProviderConformanceResult result = ProviderConformance.verify(() -> provider, CONTEXT);

        assertTrue(result.isConformant(), result.violations().toString());
        assertEquals("example/reference", result.providerId());
        assertTrue(initialized.get());
        assertTrue(closed.get());
    }

    @Test
    void closesProviderWhenInitializationFails() {
        AtomicBoolean closed = new AtomicBoolean();
        AljabrProvider provider = provider(
                () -> { throw new ProviderException("init failed"); },
                () -> closed.set(true));

        ProviderConformanceResult result = ProviderConformance.verify(() -> provider, CONTEXT);

        assertFalse(result.isConformant());
        assertTrue(result.violations().stream().anyMatch(v -> v.contains("initialize() failed")));
        assertTrue(closed.get());
    }

    @Test
    void reportsFactoryFailure() {
        ProviderConformanceResult result = ProviderConformance.verify(
                () -> { throw new IllegalStateException("construction failed"); }, CONTEXT);

        assertFalse(result.isConformant());
        assertEquals("<unavailable>", result.providerId());
        assertTrue(result.violations().get(0).contains("provider factory failed"));
    }

    @Test
    void reportsCloseFailureAfterSuccessfulInitialization() {
        AtomicBoolean initialized = new AtomicBoolean();
        AljabrProvider provider = provider(
                () -> initialized.set(true),
                () -> { throw new ProviderException("close failed"); });

        ProviderConformanceResult result = ProviderConformance.verify(() -> provider, CONTEXT);

        assertTrue(initialized.get());
        assertFalse(result.isConformant());
        assertTrue(result.violations().stream().anyMatch(v -> v.contains("close() failed")));
    }

    @Test
    void resultDefensivelyCopiesViolations() {
        var violations = new java.util.ArrayList<String>();
        violations.add("failure");

        ProviderConformanceResult result = new ProviderConformanceResult("example/provider", violations);
        violations.add("later failure");

        assertEquals(java.util.List.of("failure"), result.violations());
        assertThrows(UnsupportedOperationException.class, () -> result.violations().add("mutation"));
    }

    @Test
    void reportsFactoryReturningNull() {
        ProviderConformanceResult result = ProviderConformance.verify(() -> null, CONTEXT);

        assertFalse(result.isConformant());
        assertEquals("<null>", result.providerId());
    }

    @Test
    void reportsDescriptorFailureAndStillExercisesLifecycle() {
        AtomicBoolean closed = new AtomicBoolean();
        AljabrProvider provider = new AljabrProvider() {
            @Override public ProviderDescriptor descriptor() {
                throw new IllegalStateException("descriptor unavailable");
            }
            @Override public void close() { closed.set(true); }
        };

        ProviderConformanceResult result = ProviderConformance.verify(() -> provider, CONTEXT);

        assertFalse(result.isConformant());
        assertTrue(result.violations().stream().anyMatch(v -> v.contains("descriptor() failed")));
        assertTrue(closed.get());
    }

    @FunctionalInterface
    private interface ProviderAction {
        void run() throws Exception;
    }

    private static AljabrProvider provider(ProviderAction initialize, ProviderAction close) {
        return new AljabrProvider() {
            @Override
            public ProviderDescriptor descriptor() {
                return new ProviderDescriptor("example/reference", "1.0.0", ">=0.1.0 <1.0.0",
                        Set.of(new CapabilityDescriptor("math.vector", "1", CapabilityStability.EXPERIMENTAL)));
            }

            @Override public void initialize(ProviderContext context) throws ProviderException {
                try { initialize.run(); } catch (ProviderException exception) { throw exception; }
                catch (Exception exception) { throw new ProviderException("test initialization failed", exception); }
            }
            @Override public void close() throws ProviderException {
                try { close.run(); } catch (ProviderException exception) { throw exception; }
                catch (Exception exception) { throw new ProviderException("test close failed", exception); }
            }
        };
    }
}
