package example.host;

import java.util.Map;
import java.util.ServiceLoader;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.ApiCompatibilityRange;
import tech.kayys.aljabr.spi.provider.ProviderContext;

public final class ProviderDiscoveryDemo {
    private ProviderDiscoveryDemo() {
    }

    public static void main(String[] args) {
        ProviderContext context = new ProviderContext("0.1.0", Map.of());
        int[] discovered = {0};

        ServiceLoader.load(AljabrProvider.class).forEach(provider -> {
            try {
                var descriptor = provider.descriptor();
                if (!ApiCompatibilityRange.parse(descriptor.apiCompatibility())
                        .includes(context.aljabrVersion())) {
                    throw new IllegalStateException("External provider is incompatible with this Aljabr version");
                }
                provider.initialize(context);
                System.out.println("Discovered and initialized provider: " + descriptor.id());
                discovered[0]++;
            } finally {
                provider.close();
            }
        });

        if (discovered[0] != 1) {
            throw new IllegalStateException("Expected exactly one external provider, found " + discovered[0]);
        }
    }
}
