package tech.kayys.aljabr.spi.provider;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
/** Immutable, provider-scoped initialization context. */
public record ProviderContext(String aljabrVersion, Map<String, String> configuration) {
    public ProviderContext {
        if (aljabrVersion == null || aljabrVersion.isBlank())
            throw new IllegalArgumentException("aljabrVersion must not be blank");
        configuration = Map.copyOf(Objects.requireNonNull(configuration, "configuration"));
    }
    public Optional<String> config(String key) {
        return Optional.ofNullable(configuration.get(Objects.requireNonNull(key, "key")));
    }
    public String config(String key, String defaultValue) { return config(key).orElse(defaultValue); }
}
