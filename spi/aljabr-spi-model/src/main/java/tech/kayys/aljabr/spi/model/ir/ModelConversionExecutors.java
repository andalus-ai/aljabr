package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;

/** Discovers and selects format-neutral conversion executor providers. */
public final class ModelConversionExecutors {

    private ModelConversionExecutors() {
    }

    /** Discovers providers visible to the thread context class loader. */
    public static List<ModelConversionExecutor> discover() {
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        ServiceLoader<ModelConversionExecutor> services = loader == null
                ? ServiceLoader.load(ModelConversionExecutor.class)
                : ServiceLoader.load(ModelConversionExecutor.class, loader);
        return services.stream().map(ServiceLoader.Provider::get).toList();
    }

    /**
     * Finds the unique provider for a source/target format pair.
     *
     * @throws IllegalStateException if multiple providers claim the same pair
     */
    public static Optional<ModelConversionExecutor> find(
            List<? extends ModelConversionExecutor> executors,
            String sourceFormat,
            String targetFormat) {
        Objects.requireNonNull(executors, "executors must not be null");
        Objects.requireNonNull(sourceFormat, "sourceFormat must not be null");
        Objects.requireNonNull(targetFormat, "targetFormat must not be null");
        List<ModelConversionExecutor> matches = executors.stream()
                .filter(Objects::nonNull)
                .filter(executor -> executor.supports(sourceFormat, targetFormat))
                .map(executor -> (ModelConversionExecutor) executor)
                .toList();
        if (matches.size() > 1) {
            throw new IllegalStateException("Multiple conversion executors support "
                    + sourceFormat + " -> " + targetFormat);
        }
        return matches.stream().findFirst();
    }

    /** Finds a provider using ServiceLoader discovery. */
    public static Optional<ModelConversionExecutor> find(
            String sourceFormat, String targetFormat) {
        return find(discover(), sourceFormat, targetFormat);
    }
}
