package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import tech.kayys.aljabr.spi.model.ir.ArchitectureConfiguration;
import tech.kayys.aljabr.spi.model.ir.ArchitectureDescriptor;
import tech.kayys.aljabr.spi.model.ir.ArchitectureFamily;
import tech.kayys.aljabr.spi.model.ir.ArchitectureKind;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Versioned, format-neutral envelope for the typed AMIR architecture layer.
 *
 * <p>The envelope is stored as a single string metadata value so both GGUF
 * and Safetensors can carry the exact same representation. It is deliberately
 * independent of either container's native metadata conventions.
 */
final class AmirArchitectureDescriptorCodec {
    static final String KEY = "aljabr.amir.architecture.v1";
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private AmirArchitectureDescriptorCodec() { }

    static String encode(ArchitectureDescriptor descriptor) {
        JsonObject root = new JsonObject();
        root.addProperty("id", descriptor.id());
        root.addProperty("familyId", descriptor.family().id());
        root.addProperty("familyKind", descriptor.family().kind().name());

        JsonObject properties = new JsonObject();
        descriptor.configuration().properties().forEach((key, value) ->
                properties.add(key, encodeValue(value)));
        root.add("properties", properties);
        root.add("configurationExtensions", stringMap(descriptor.configuration().extensions()));

        JsonArray capabilities = new JsonArray();
        descriptor.capabilities().stream().sorted().forEach(capabilities::add);
        root.add("capabilities", capabilities);
        root.add("extensions", stringMap(descriptor.extensions()));
        return GSON.toJson(root);
    }

    static ArchitectureDescriptor decode(String encoded) {
        try {
            JsonObject root = JsonParser.parseString(encoded).getAsJsonObject();
            String id = requiredString(root, "id");
            ArchitectureFamily family = new ArchitectureFamily(
                    requiredString(root, "familyId"),
                    ArchitectureKind.valueOf(requiredString(root, "familyKind")));

            Map<String, ConfigurationValue> properties = new LinkedHashMap<>();
            JsonObject encodedProperties = requiredObject(root, "properties");
            encodedProperties.entrySet().forEach(entry ->
                    properties.put(entry.getKey(), decodeValue(entry.getValue())));

            Set<String> capabilities = new LinkedHashSet<>();
            JsonArray encodedCapabilities = requiredArray(root, "capabilities");
            encodedCapabilities.forEach(value -> capabilities.add(value.getAsString()));

            return new ArchitectureDescriptor(id, family,
                    new ArchitectureConfiguration(properties,
                            decodeStringMap(requiredObject(root, "configurationExtensions"))),
                    capabilities, decodeStringMap(requiredObject(root, "extensions")));
        } catch (RuntimeException invalid) {
            throw new IllegalArgumentException("Invalid AMIR architecture descriptor envelope", invalid);
        }
    }

    static boolean isEncodedKey(String key) {
        return KEY.equals(key);
    }

    private static JsonObject encodeValue(ConfigurationValue value) {
        JsonObject encoded = new JsonObject();
        if (value instanceof ConfigurationValue.StringValue v) {
            encoded.addProperty("type", "string");
            encoded.addProperty("value", v.value());
        } else if (value instanceof ConfigurationValue.IntegerValue v) {
            encoded.addProperty("type", "integer");
            encoded.addProperty("value", Long.toString(v.value()));
        } else if (value instanceof ConfigurationValue.DecimalValue v) {
            encoded.addProperty("type", "decimal");
            encoded.addProperty("value", v.value().toPlainString());
        } else if (value instanceof ConfigurationValue.BooleanValue v) {
            encoded.addProperty("type", "boolean");
            encoded.addProperty("value", v.value());
        } else if (value instanceof ConfigurationValue.StringListValue v) {
            encoded.addProperty("type", "string-list");
            JsonArray values = new JsonArray();
            v.value().forEach(values::add);
            encoded.add("value", values);
        } else if (value instanceof ConfigurationValue.IntegerListValue v) {
            encoded.addProperty("type", "integer-list");
            JsonArray values = new JsonArray();
            v.value().forEach(number -> values.add(Long.toString(number)));
            encoded.add("value", values);
        } else {
            throw new IllegalArgumentException("Unsupported AMIR configuration value: " + value.getClass());
        }
        return encoded;
    }

    private static ConfigurationValue decodeValue(JsonElement element) {
        JsonObject value = element.getAsJsonObject();
        return switch (requiredString(value, "type")) {
            case "string" -> new ConfigurationValue.StringValue(requiredString(value, "value"));
            case "integer" -> new ConfigurationValue.IntegerValue(
                    Long.parseLong(requiredString(value, "value")));
            case "decimal" -> new ConfigurationValue.DecimalValue(
                    new BigDecimal(requiredString(value, "value")));
            case "boolean" -> new ConfigurationValue.BooleanValue(
                    requiredBoolean(value, "value"));
            case "string-list" -> new ConfigurationValue.StringListValue(
                    stringList(requiredArray(value, "value")));
            case "integer-list" -> new ConfigurationValue.IntegerListValue(
                    integerList(requiredArray(value, "value")));
            default -> throw new IllegalArgumentException("Unknown architecture configuration value type");
        };
    }

    private static JsonObject stringMap(Map<String, String> values) {
        JsonObject object = new JsonObject();
        values.forEach(object::addProperty);
        return object;
    }

    private static Map<String, String> decodeStringMap(JsonObject object) {
        Map<String, String> values = new LinkedHashMap<>();
        object.entrySet().forEach(entry -> values.put(entry.getKey(), entry.getValue().getAsString()));
        return values;
    }

    private static List<String> stringList(JsonArray values) {
        List<String> result = new ArrayList<>(values.size());
        values.forEach(value -> result.add(value.getAsString()));
        return result;
    }

    private static List<Long> integerList(JsonArray values) {
        List<Long> result = new ArrayList<>(values.size());
        values.forEach(value -> result.add(Long.parseLong(value.getAsString())));
        return result;
    }

    private static String requiredString(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new IllegalArgumentException("Missing or invalid string field: " + key);
        }
        return value.getAsString();
    }

    private static boolean requiredBoolean(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean()) {
            throw new IllegalArgumentException("Missing or invalid boolean field: " + key);
        }
        return value.getAsBoolean();
    }

    private static JsonObject requiredObject(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonObject()) {
            throw new IllegalArgumentException("Missing or invalid object field: " + key);
        }
        return value.getAsJsonObject();
    }

    private static JsonArray requiredArray(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonArray()) {
            throw new IllegalArgumentException("Missing or invalid array field: " + key);
        }
        return value.getAsJsonArray();
    }
}
