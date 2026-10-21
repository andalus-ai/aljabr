package tech.kayys.aljabr.spi.registry;

import tech.kayys.aljabr.core.model.ModelFormat;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
/**
 * Immutable record representing modelentry data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelEntry(
        String modelId,
        String name,
        ModelFormat format,
        Path physicalPath,
        long sizeBytes,
        String provider,
        Map<String, String> metadata,
        Instant registeredAt) {

    public ModelEntry {
        modelId = modelId == null ? "" : modelId;
        name = name == null ? "" : name;
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
        registeredAt = registeredAt == null ? Instant.now() : registeredAt;
    }

    public Path path() {
        return physicalPath;
    }

    public static Builder builder() {
        return new Builder();
    }/**
 * Builder for constructing complex  objects.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */


    public static class Builder {
        private String modelId;
        private String name;
        private ModelFormat format;
        private Path physicalPath;
        private long sizeBytes;
        private String provider;
        private Map<String, String> metadata;
        private Instant registeredAt;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public Builder physicalPath(Path physicalPath) {
            this.physicalPath = physicalPath;
            return this;
        }

        public Builder sizeBytes(long sizeBytes) {
            this.sizeBytes = sizeBytes;
            return this;
        }

        public Builder provider(String provider) {
            this.provider = provider;
            return this;
        }

        public Builder metadata(Map<String, String> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder registeredAt(Instant registeredAt) {
            this.registeredAt = registeredAt;
            return this;
        }

        public ModelEntry build() {
            return new ModelEntry(modelId, name, format, physicalPath, sizeBytes, provider, metadata, registeredAt);
        }
    }
}
