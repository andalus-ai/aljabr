package tech.kayys.aljabr.spi.testkit;

import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderDescriptor;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Reusable checks for authors of Aljabr providers.
 *
 * <p>Checks descriptor availability and the basic initialize/close lifecycle only.
 * Capability-specific correctness checks belong in the conformance suites for those
 * capability SPIs, once their operation contracts exist.</p>
 */
public final class ProviderConformance {
    private ProviderConformance() {
    }

    /**
     * Creates a fresh provider, checks its descriptor, then exercises initialize and close.
     * The provider is closed even when initialization fails.
     *
     * @param factory creates a fresh provider instance for this check
     * @param context initialization context
     * @return an immutable report; provider failures are captured as violations
     */
    public static ProviderConformanceResult verify(
            Supplier<? extends AljabrProvider> factory,
            ProviderContext context) {
        Objects.requireNonNull(factory, "factory");
        Objects.requireNonNull(context, "context");

        var violations = new ArrayList<String>();
        AljabrProvider provider;
        try {
            provider = factory.get();
        } catch (RuntimeException exception) {
            return new ProviderConformanceResult("<unavailable>",
                    java.util.List.of("provider factory failed: " + describe(exception)));
        }

        if (provider == null) {
            return new ProviderConformanceResult("<null>",
                    java.util.List.of("provider factory returned null"));
        }

        String providerId = "<unknown>";
        try {
            ProviderDescriptor descriptor = provider.descriptor();
            if (descriptor == null) {
                violations.add("descriptor() returned null");
            } else {
                providerId = descriptor.id();
            }
        } catch (RuntimeException exception) {
            violations.add("descriptor() failed: " + describe(exception));
        }

        try {
            provider.initialize(context);
        } catch (Exception exception) {
            violations.add("initialize() failed: " + describe(exception));
        } finally {
            try {
                provider.close();
            } catch (Exception exception) {
                violations.add("close() failed: " + describe(exception));
            }
        }

        return new ProviderConformanceResult(providerId, violations);
    }

    private static String describe(Exception exception) {
        String message = exception.getMessage();
        return exception.getClass().getSimpleName()
                + (message == null || message.isBlank() ? "" : ": " + message);
    }
}
