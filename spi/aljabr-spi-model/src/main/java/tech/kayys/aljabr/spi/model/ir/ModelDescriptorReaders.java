package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;

/** Discovers and selects format-specific AMIR descriptor readers. */
public final class ModelDescriptorReaders {

    private ModelDescriptorReaders() {
    }

    /** Discovers readers visible to the thread context class loader. */
    public static List<ModelDescriptorReader> discover() {
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        ServiceLoader<ModelDescriptorReader> services = loader == null
                ? ServiceLoader.load(ModelDescriptorReader.class)
                : ServiceLoader.load(ModelDescriptorReader.class, loader);
        return services.stream().map(ServiceLoader.Provider::get).toList();
    }

    /**
     * Finds the unique reader for a source format.
     *
     * @throws IllegalStateException when more than one reader claims the format
     */
    public static Optional<ModelDescriptorReader> find(
            List<? extends ModelDescriptorReader> readers, String sourceFormat) {
        Objects.requireNonNull(readers, "readers must not be null");
        if (sourceFormat == null || sourceFormat.isBlank()) {
            throw new IllegalArgumentException("sourceFormat must not be blank");
        }
        List<ModelDescriptorReader> matches = readers.stream()
                .filter(Objects::nonNull)
                .filter(reader -> reader.supports(sourceFormat))
                .map(reader -> (ModelDescriptorReader) reader)
                .toList();
        if (matches.size() > 1) {
            throw new IllegalStateException(
                    "Multiple AMIR descriptor readers support format " + sourceFormat
                            + ": " + matches.stream().map(r -> r.getClass().getName()).toList());
        }
        return matches.stream().findFirst();
    }
}
