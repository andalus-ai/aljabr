package example.host;

import java.util.Map;
import java.util.ServiceLoader;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.ProviderContext;

public final class ProviderDiscoveryDemo {
    private ProviderDiscoveryDemo() {
    }

    public static void main(String[] args) {
        ProviderContext context = new ProviderContext("0.1.0", Map.of());

        long discovered = ServiceLoader.load(AljabrProvider.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .peek(provider -> {
                    try {
                        provider.initialize(context);
                        System.out.println(provider.descriptor().id()
                                + " -> " + ((example.provider.ReferenceGreetingProvider) provider)
                                        .greet("Aljabr"));
                    } finally {
                        provider.close();
                    }
                })
                .count();

        if (discovered != 1) {
            throw new IllegalStateException("Expected exactly one external provider, found " + discovered);
        }
    }
}
