package tech.kayys.aljabr.gguf.loader;

import io.quarkus.runtime.annotations.RegisterForReflection;

/**
 * Metadata and offset for a single tensor within a GGUF file.
 * @author bhangun
 */
@RegisterForReflection
public record GGUFTensorInfo(
    String name,
    long[] shape,
    int typeId,
    long offset,
    long sizeInBytes
) {}
