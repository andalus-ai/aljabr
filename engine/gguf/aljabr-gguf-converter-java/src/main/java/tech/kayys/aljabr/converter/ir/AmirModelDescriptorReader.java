package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.converter.java.gguf.SafetensorsReader;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptorReader;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Built-in metadata-only AMIR readers for GGUF and Safetensors files.
 *
 * <p>Both readers retain payloads as file ranges. GGUF dimensions are kept in
 * their source ggml order; conversion executors are responsible for the
 * format-specific logical dimension normalization.
 */
public final class AmirModelDescriptorReader implements ModelDescriptorReader {

    public static final String GGUF = "GGUF";
    public static final String SAFETENSORS = "SAFETENSORS";

    @Override
    public String format() {
        throw new UnsupportedOperationException(
                "This provider supports multiple formats; use supports(Path) dispatch");
    }

    @Override
    public boolean supports(String sourceFormat) {
        return GGUF.equalsIgnoreCase(sourceFormat)
                || SAFETENSORS.equalsIgnoreCase(sourceFormat);
    }

    @Override
    public ModelDescriptor read(Path source) throws IOException {
        Path file = requireFile(source);
        String name = file.getFileName() == null ? file.toString() : file.getFileName().toString();
        String lower = name.toLowerCase(java.util.Locale.ROOT);
        if (lower.endsWith(".gguf")) {
            return readGguf(file, name);
        }
        if (lower.endsWith(".safetensors")) {
            return readSafetensors(file, name);
        }
        throw new IllegalArgumentException("Cannot infer model format from file extension: " + name);
    }

    private ModelDescriptor readGguf(Path file, String name) throws IOException {
        Map<String, Object> rawMetadata;
        List<TensorDescriptor> tensors = new ArrayList<>();
        long fileSize = Files.size(file);
        long dataStart;

        try (GGUFReader reader = new GGUFReader(file)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), reader.arena());
            rawMetadata = model.metadata();
            dataStart = model.dataStart();
            for (GGUFTensorInfo tensor : model.tensors()) {
                GgmlType type = GgmlType.fromId(tensor.typeId());
                long absoluteOffset = Math.addExact(dataStart, tensor.offset());
                validateRange(absoluteOffset, tensor.sizeInBytes(), fileSize, tensor.name());
                List<Long> shape = java.util.Arrays.stream(tensor.shape()).boxed().toList();
                PayloadReference payload = new PayloadReference(
                        file.toUri(), name, absoluteOffset, tensor.sizeInBytes(),
                        PayloadReference.OffsetBasis.CONTAINER);
                Map<String, String> extensions = Map.of(
                        "gguf.type-id", Integer.toString(tensor.typeId()),
                        "gguf.dimension-order", "ggml");
                tensors.add(new TensorDescriptor(
                        tensor.name(), type.label, shape, "unknown", null, payload, extensions));
            }
        } catch (ArithmeticException | IllegalArgumentException e) {
            throw new IOException("Invalid GGUF model structure: " + file, e);
        }

        Map<String, String> metadata = new LinkedHashMap<>();
        Map<String, String> extensions = new LinkedHashMap<>();
        rawMetadata.forEach((key, value) -> {
            String rendered = renderMetadata(value);
            metadata.put(key, rendered);
            if (!(value instanceof String)) {
                extensions.put("gguf.metadata." + key, rendered);
            }
        });
        String architecture = metadata.getOrDefault("general.architecture", "unknown");
        return new ModelDescriptor(ModelIdentity.of(name), architecture, GGUF,
                metadata, extensions, tensors);
    }

    private ModelDescriptor readSafetensors(Path file, String name) throws IOException {
        Map<String, String> metadata;
        List<TensorDescriptor> tensors = new ArrayList<>();
        long fileSize = Files.size(file);

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            metadata = new LinkedHashMap<>(reader.globalMeta());
            for (SafetensorsReader.TensorEntry tensor : reader.tensors().values()) {
                SafetensorDType type = SafetensorDType.fromJson(tensor.dtype());
                long length = tensor.byteSize();
                if (tensor.dataStart() < 0 || length < 0) {
                    throw new IOException("Invalid Safetensors payload range for " + tensor.name());
                }
                long expected = type.totalBytes(tensor.shape());
                if (expected < 0 || expected != length) {
                    throw new IOException("Safetensors payload size does not match dtype/shape for "
                            + tensor.name() + ": expected " + expected + ", found " + length);
                }
                // DATA_SECTION offsets deliberately remain relative to the Safetensors data area.
                PayloadReference payload = new PayloadReference(
                        file.toUri(), name, tensor.dataStart(), length,
                        PayloadReference.OffsetBasis.DATA_SECTION);
                tensors.add(new TensorDescriptor(
                        tensor.name(), type.jsonName(),
                        java.util.Arrays.stream(tensor.shape()).boxed().toList(),
                        "contiguous", null, payload));
            }
            long dataLength = fileSize - (fileSize - reader.tensors().values().stream()
                    .mapToLong(SafetensorsReader.TensorEntry::dataEnd).max().orElse(0L));
            if (dataLength < 0) {
                throw new IOException("Invalid Safetensors data section");
            }
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid Safetensors model structure: " + file, e);
        }

        String architecture = metadata.getOrDefault("format", "unknown");
        return new ModelDescriptor(ModelIdentity.of(name), architecture, SAFETENSORS,
                metadata, Map.of(), tensors);
    }

    private static Path requireFile(Path source) throws IOException {
        if (source == null) {
            throw new IllegalArgumentException("source must not be null");
        }
        Path file = source.toAbsolutePath().normalize();
        if (!Files.isRegularFile(file) || !Files.isReadable(file)) {
            throw new IOException("Model source is not a readable regular file: " + file);
        }
        return file;
    }

    private static void validateRange(long offset, long length, long fileSize, String tensor)
            throws IOException {
        final long end;
        try {
            end = Math.addExact(offset, length);
        } catch (ArithmeticException e) {
            throw new IOException("GGUF payload range overflows for tensor " + tensor, e);
        }
        if (end > fileSize) {
            throw new IOException("GGUF payload exceeds source file for tensor " + tensor);
        }
    }

    private static String renderMetadata(Object value) {
        if (value instanceof List<?> values) {
            return values.stream().map(AmirModelDescriptorReader::renderMetadata)
                    .collect(java.util.stream.Collectors.joining(",", "[", "]"));
        }
        return String.valueOf(value);
    }
}
