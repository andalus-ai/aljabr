package example.provider;

import java.util.Map;
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
        if (!DESCRIPTOR.id().equals("example/greeting")) {
            throw new IllegalStateException("unexpected provider descriptor");
        }
        initialized = true;
    }

    public String greet(String name) {
        if (!initialized || closed) {
            throw new IllegalStateException("provider is not active");
        }
        return "Hello, " + name + "!";
    }

    @Override
    public void close() {
        closed = true;
        initialized = false;
    }
}
