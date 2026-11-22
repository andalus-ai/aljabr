package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import tech.kayys.aljabr.spi.model.ir.MetadataValue;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Encodes AMIR typed metadata into Safetensors' string-only metadata map.
 *
 * <p>Reserved keys carry a versioned, type-tagged representation. Values are
 * encoded as strings inside the envelope so unsigned 64-bit bit patterns and
 * non-finite floating-point values survive JSON serialization exactly.
 */
final class AmirTypedMetadataCodec {

    static final String PREFIX = "aljabr.amir.typed-metadata.v1.";

    private AmirTypedMetadataCodec() {
    }

    static Map<String, String> encode(Map<String, MetadataValue> metadata) {
        Map<String, String> encoded = new LinkedHashMap<>();
        metadata.forEach((key, value) -> encoded.put(keyFor(key), encodeValue(value)));
        return Map.copyOf(encoded);
    }

    static Map<String, MetadataValue> decode(Map<String, String> metadata) {
        Map<String, MetadataValue> decoded = new LinkedHashMap<>();
        metadata.forEach((key, value) -> {
            if (key.startsWith(PREFIX)) {
                String originalKey = keyFrom(key);
                final MetadataValue decodedValue;
                try {
                    decodedValue = decodeValue(value);
                } catch (RuntimeException invalid) {
                    throw new IllegalArgumentException(
                            "Invalid AMIR typed metadata value for key: " + originalKey, invalid);
                }
                if (decoded.putIfAbsent(originalKey, decodedValue) != null) {
                    throw new IllegalArgumentException("Duplicate typed metadata key: " + originalKey);
                }
            }
        });
        return Map.copyOf(decoded);
    }

    static Map<String, String> withoutEncodedValues(Map<String, String> metadata) {
        Map<String, String> ordinary = new LinkedHashMap<>();
        metadata.forEach((key, value) -> {
            if (!key.startsWith(PREFIX)) ordinary.put(key, value);
        });
        return Map.copyOf(ordinary);
    }

    private static String keyFor(String key) {
        return PREFIX + Base64.getUrlEncoder().withoutPadding()
                .encodeToString(key.getBytes(StandardCharsets.UTF_8));
    }

    private static String keyFrom(String encodedKey) {
        try {
            String suffix = encodedKey.substring(PREFIX.length());
            String key = new String(Base64.getUrlDecoder().decode(suffix), StandardCharsets.UTF_8);
            if (key.isBlank()) throw new IllegalArgumentException("Typed metadata key must not be blank");
            return key;
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Invalid AMIR typed metadata key: " + encodedKey, invalid);
        }
    }

    private static String encodeValue(MetadataValue value) {
        JsonObject envelope = new JsonObject();
        if (value instanceof MetadataValue.Scalar scalar) {
            envelope.addProperty("kind", scalar.kind().name());
            envelope.addProperty("value", scalarText(scalar));
        } else {
            MetadataValue.Array array = (MetadataValue.Array) value;
            envelope.addProperty("kind", "ARRAY");
            envelope.addProperty("elementKind", array.elementKind().name());
            JsonArray values = new JsonArray();
            array.values().forEach(element -> values.add(scalarText(element)));
            envelope.add("values", values);
        }
        return envelope.toString();
    }

    private static MetadataValue decodeValue(String encoded) {
        JsonObject envelope = JsonParser.parseString(encoded).getAsJsonObject();
        String kind = envelope.get("kind").getAsString();
        if ("ARRAY".equals(kind)) {
            MetadataValue.Kind elementKind = MetadataValue.Kind.valueOf(
                    envelope.get("elementKind").getAsString());
            JsonArray values = envelope.getAsJsonArray("values");
            return new MetadataValue.Array(elementKind, values.asList().stream()
                    .map(value -> scalar(elementKind, value.getAsString())).toList());
        }
        return scalar(MetadataValue.Kind.valueOf(kind), envelope.get("value").getAsString());
    }

    private static String scalarText(MetadataValue.Scalar scalar) {
        return switch (scalar.kind()) {
            case FLOAT32 -> Float.toHexString((Float) scalar.value());
            case FLOAT64 -> Double.toHexString((Double) scalar.value());
            default -> String.valueOf(scalar.value());
        };
    }

    private static MetadataValue.Scalar scalar(MetadataValue.Kind kind, String value) {
        Object decoded = switch (kind) {
            case UINT8, UINT16, UINT32, UINT64, INT64 -> Long.parseLong(value);
            case INT8, INT16, INT32 -> Integer.parseInt(value);
            case FLOAT32 -> Float.valueOf(value);
            case FLOAT64 -> Double.valueOf(value);
            case BOOL -> {
                if (!"true".equals(value) && !"false".equals(value)) {
                    throw new IllegalArgumentException("Invalid boolean metadata value: " + value);
                }
                yield Boolean.valueOf(value);
            }
            case STRING -> value;
        };
        return new MetadataValue.Scalar(kind, decoded);
    }
}
