package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

/**
 * Represents the location of a model artifact, including its URI and additional
 * metadata.
 * @author bhangun
 */
public record ArtifactLocation(
        String uri,
        String checksum,
        Long size,
        String contentType) {

    public ArtifactLocation {
        if (uri == null || uri.isBlank()) {
            throw new IllegalArgumentException("URI cannot be null or blank");
        }
    }
}