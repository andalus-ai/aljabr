package example.provider;

import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.CapabilityDescriptor;
import tech.kayys.aljabr.spi.provider.CapabilityStability;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDescriptor;

/**
 * Example provider implemented outside the Aljabr repository.
 *
 * This implementation intentionally uses only the public provider SPI.
 */
public final class ReferenceGreetingProvider implements AljabrProvider {
    private static final ProviderDescriptor DESCRIPTOR = new ProviderDescriptor(
            "example/greeting",
            "1.0.0",
            ">=0.1.0 <1.0.0",
            java.util.Set.of(new CapabilityDescriptor(
                    "example.greeting",
                    "1.0.0",
                    CapabilityStability.PREVIEW)));

    private boolean initialized;
    private boolean closed;

    @Override
    public ProviderDescriptor descriptor() {
        return DESCRIPTOR;
    }

    @Override
    public void initialize(ProviderContext context) {
        if (closed) {
            throw new IllegalStateException("provider is closed");
        }
        if (initialized) {
            throw new IllegalStateException("provider is already initialized");
        }
        initialized = true;
    }

    @Override
    public void close() {
        closed = true;
        initialized = false;
    }
}
