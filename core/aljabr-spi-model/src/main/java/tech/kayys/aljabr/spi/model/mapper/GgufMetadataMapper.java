package tech.kayys.aljabr.spi.model.mapper;

import tech.kayys.aljabr.spi.model.ModelConfig;

import java.util.Map;

/**
 * Maps a raw GGUF metadata map (key → value) to a {@link ModelConfig}.
 *
 * <p>This thin wrapper delegates to {@link ModelConfig#fromGgufMetadata(Map)},
 * providing a stable mapper interface so that callers in other modules
 * (e.g. {@code aljabr-gguf-core}) do not depend directly on the static
 * factory method and can be swapped or mocked in tests.</p>
 * @author bhangun
 */
public class GgufMetadataMapper {

    /**
     * Derives a {@link ModelConfig} from the given GGUF metadata map.
     *
     * @param metadata key-value pairs from {@code GGUFModel.metadata()}
     * @return a {@code ModelConfig} with fields populated from the metadata,
     *         or an empty default config if {@code metadata} is null/empty
     */
    public ModelConfig fromGgufMetadata(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return new ModelConfig();
        }
        return ModelConfig.fromGgufMetadata(metadata);
    }
}
