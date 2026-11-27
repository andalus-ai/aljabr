package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class GgufBackendRegistrationTest {

    @Test
    void retirementRejectsNewLeasesAndClosesAfterFinalLeaseExactlyOnce() {
        AtomicInteger closes = new AtomicInteger();
        GgufBackend backend = new TestBackend();
        GgufBackendRegistration registration =
                new GgufBackendRegistration(backend, closes::incrementAndGet);

        GgufBackendRegistration.Lease first = registration.acquire();
        GgufBackendRegistration.Lease second = registration.acquire();

        registration.retire();
        registration.retire();

        assertEquals(0, closes.get());
        assertSame(backend, first.backend());
        assertThrows(IllegalStateException.class, registration::acquire);

        first.close();
        first.close();
        assertEquals(0, closes.get());

        second.close();
        second.close();
        assertEquals(1, closes.get());
    }

    @Test
    void retirementWithoutActiveLeasesClosesImmediatelyAndOnlyOnce() {
        AtomicInteger closes = new AtomicInteger();
        GgufBackendRegistration registration =
                new GgufBackendRegistration(new TestBackend(), closes::incrementAndGet);

        registration.retire();
        registration.retire();

        assertEquals(1, closes.get());
        assertThrows(IllegalStateException.class, registration::acquire);
    }

    @Test
    void deferredCloseFailureIsPropagatedAfterLeaseAccountingCompletes() {
        IllegalStateException closeFailure = new IllegalStateException("close failed");
        AtomicInteger closes = new AtomicInteger();
        GgufBackendRegistration registration = new GgufBackendRegistration(
                new TestBackend(),
                () -> {
                    closes.incrementAndGet();
                    throw closeFailure;
                });
        GgufBackendRegistration.Lease lease = registration.acquire();

        registration.retire();

        assertSame(closeFailure, assertThrows(IllegalStateException.class, lease::close));
        assertEquals(1, closes.get());
        assertDoesNotThrow(lease::close, "lease close is idempotent even when provider cleanup fails");
        assertEquals(1, closes.get());
    }

    private static final class TestBackend implements GgufBackend {
        @Override
        public String name() {
            return "lease-test";
        }

        @Override
        public tech.kayys.aljabr.spi.inference.InferenceResponse execute(
                tech.kayys.aljabr.spi.inference.InferenceRequest request) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void close() {
        }
    }
}
