package tech.kayys.aljabr.safetensor.spi;

import tech.kayys.aljabr.spi.model.ModalityType;

/**
 * Represents a single unit of fused input.
 * 
 * <p>Contains the embedding vector, its original modality, and its 
 * assigned sequence position.
 * @author bhangun
 */
public record FusedToken(
    float[] embedding,
    ModalityType modality,
    int position
) {}
