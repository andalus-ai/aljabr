package tech.kayys.aljabr.gguf.loader;

import io.quarkus.runtime.annotations.RegisterForReflection;

/**
 * Parsed GGUF tensor file-layout metadata.
 *
 * <p>{@code offset} is relative to the GGUF tensor-data section and
 * {@code sizeInBytes} is the serialized payload size. This parser record
 * deliberately does not expose a runtime {@code MemorySegment} or resolve
 * the type ID to a backend/runtime type. Use {@code GgufTensorDescriptor}
 * for the public structural inspection contract.
 *
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
