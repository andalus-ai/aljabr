package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.converter.java.gguf.SafetensorsReader;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.spi.model.ir.ArchitectureDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptorReader;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.MetadataValue;
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
    public java.util.Set<String> formats() {
        return java.util.Set.of(GGUF, SAFETENSORS);
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
        Map<String, tech.kayys.aljabr.gguf.core.GgufMetaValue> rawTypedMetadata;
        List<TensorDescriptor> tensors = new ArrayList<>();
        long fileSize = Files.size(file);
        long dataStart;

        try (GGUFReader reader = new GGUFReader(file)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), reader.arena());
            rawMetadata = model.metadata();
            rawTypedMetadata = model.typedMetadata();
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
        for (Map.Entry<String, Object> entry : rawMetadata.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            String rendered = rawTypedMetadata.containsKey(key)
                    ? renderGgufMetadata(rawTypedMetadata.get(key)) : renderMetadata(value);
            metadata.put(key, rendered);
        }
        Map<String, String> modelLayerMetadata = new LinkedHashMap<>(metadata);
        ArchitectureDescriptor architectureDescriptor = decodeArchitectureDescriptor(metadata);
        List<TensorDescriptor> typedTensors = AmirTensorTypeCodec.restore(metadata, tensors);
        AmirSourceMetadataCodec.Decoded sourceAttributes =
                AmirSourceMetadataCodec.decode(metadata, typedTensors);
        metadata = new LinkedHashMap<>(sourceAttributes.metadata());
        tensors = new ArrayList<>(sourceAttributes.tensors());
        String architecture = metadata.getOrDefault("general.architecture", "unknown");
        Map<String, MetadataValue> typedMetadata = new LinkedHashMap<>();
        rawTypedMetadata.forEach((key, value) -> {
            if (!(value instanceof tech.kayys.aljabr.gguf.core.GgufMetaValue.StringVal)) {
                typedMetadata.put(key, toMetadataValue(value));
            }
        });
        ModelDescriptor descriptor = new ModelDescriptor(ModelIdentity.of(name), architecture, GGUF,
                metadata, sourceAttributes.extensions(), tensors).withTypedMetadata(typedMetadata);
        if (architectureDescriptor != null) descriptor = descriptor.withArchitectureDescriptor(architectureDescriptor);
        return AmirModelLayerCodec.restore(modelLayerMetadata, descriptor);
    }

    private ModelDescriptor readSafetensors(Path file, String name) throws IOException {
        Map<String, String> metadata;
        Map<String, String> storedMetadata;
        Map<String, MetadataValue> typedMetadata;
        List<TensorDescriptor> tensors = new ArrayList<>();
        long fileSize = Files.size(file);

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            storedMetadata = new LinkedHashMap<>(reader.globalMeta());
            typedMetadata = AmirTypedMetadataCodec.decode(storedMetadata);
            metadata = new LinkedHashMap<>(AmirTypedMetadataCodec.withoutEncodedValues(storedMetadata));
            for (SafetensorsReader.TensorEntry tensor : reader.tensors().values()) {
                SafetensorDType type = SafetensorDType.fromJson(tensor.dtype());
                long length = tensor.byteSize();
                if (tensor.dataStart() < 0 || length < 0
                        || Math.addExact(tensor.dataEnd(), reader.dataOffset()) > fileSize) {
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
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new IOException("Invalid Safetensors model structure: " + file, e);
        }

        Map<String, String> modelLayerMetadata = new LinkedHashMap<>(storedMetadata);
        ArchitectureDescriptor architectureDescriptor = decodeArchitectureDescriptor(storedMetadata);
        List<TensorDescriptor> typedTensors = AmirTensorTypeCodec.restore(storedMetadata, tensors);
        AmirSourceMetadataCodec.Decoded sourceAttributes =
                AmirSourceMetadataCodec.decode(storedMetadata, typedTensors);
        metadata = new LinkedHashMap<>(sourceAttributes.metadata());
        tensors = new ArrayList<>(sourceAttributes.tensors());
        String architecture = metadata.getOrDefault("general.architecture",
                metadata.getOrDefault("aljabr.source.architecture", "unknown"));
        ModelDescriptor descriptor = new ModelDescriptor(ModelIdentity.of(name), architecture, SAFETENSORS,
                metadata, sourceAttributes.extensions(), tensors).withTypedMetadata(typedMetadata);
        if (architectureDescriptor != null) descriptor = descriptor.withArchitectureDescriptor(architectureDescriptor);
        return AmirModelLayerCodec.restore(modelLayerMetadata, descriptor);
    }

    private static ArchitectureDescriptor decodeArchitectureDescriptor(Map<String, String> metadata) {
        String encoded = metadata.remove(AmirArchitectureDescriptorCodec.KEY);
        return encoded == null ? null : AmirArchitectureDescriptorCodec.decode(encoded);
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


    private static MetadataValue toMetadataValue(tech.kayys.aljabr.gguf.core.GgufMetaValue value) {
        return switch (value) {
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt8Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.UINT8, (long) v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int8Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.INT8, (int) v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt16Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.UINT16, (long) v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int16Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.INT16, (int) v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt32Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.UINT32, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int32Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.INT32, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Float32Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.FLOAT32, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.BoolVal v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.BOOL, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.StringVal v -> MetadataValue.string(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt64Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.UINT64, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int64Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.INT64, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Float64Val v ->
                    new MetadataValue.Scalar(MetadataValue.Kind.FLOAT64, v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.ArrayVal v ->
                    new MetadataValue.Array(MetadataValue.Kind.valueOf(v.elementType().name()),
                            v.elements().stream().map(element -> (MetadataValue.Scalar) toMetadataValue(element)).toList());
        };
    }

    private static String renderGgufMetadata(tech.kayys.aljabr.gguf.core.GgufMetaValue value) {
        return switch (value) {
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt8Val v -> Short.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int8Val v -> Byte.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt16Val v -> Integer.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int16Val v -> Short.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt32Val v -> Long.toUnsignedString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int32Val v -> Integer.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Float32Val v -> Float.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.BoolVal v -> Boolean.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.StringVal v -> v.value();
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.UInt64Val v -> Long.toUnsignedString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Int64Val v -> Long.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.Float64Val v -> Double.toString(v.value());
            case tech.kayys.aljabr.gguf.core.GgufMetaValue.ArrayVal v ->
                    v.elements().stream().map(AmirModelDescriptorReader::renderGgufMetadata)
                            .collect(java.util.stream.Collectors.joining(",", "[", "]"));
        };
    }

    private static String renderMetadata(Object value) {
        if (value instanceof List<?> values) {
            return values.stream().map(AmirModelDescriptorReader::renderMetadata)
                    .collect(java.util.stream.Collectors.joining(",", "[", "]"));
        }
        return String.valueOf(value);
    }
}
