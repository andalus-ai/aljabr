package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import tech.kayys.aljabr.spi.model.ir.AssetCollection;
import tech.kayys.aljabr.spi.model.ir.AssetReference;
import tech.kayys.aljabr.spi.model.ir.ComponentCollection;
import tech.kayys.aljabr.spi.model.ir.ComponentGraphBinding;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;
import tech.kayys.aljabr.spi.model.ir.ModelComponent;
import tech.kayys.aljabr.spi.model.ir.ModelComponentKind;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;

/** Versioned, format-neutral envelopes for AMIR model assets and components. */
final class AmirModelLayerCodec {
    static final String ASSETS_KEY = "aljabr.amir.assets.v1";
    static final String COMPONENTS_KEY = "aljabr.amir.components.v1";
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private AmirModelLayerCodec() { }

    static Map<String, String> encode(ModelDescriptor model) {
        Map<String, String> result = new LinkedHashMap<>();
        model.assetCollection().ifPresent(collection -> result.put(ASSETS_KEY, encodeAssets(collection)));
        model.componentCollection().filter(AmirModelLayerCodec::hasNoGraphBindings).ifPresent(collection ->
                result.put(COMPONENTS_KEY, encodeComponents(collection)));
        return Map.copyOf(result);
    }

    static ModelDescriptor restore(Map<String, String> metadata, ModelDescriptor model) {
        String assets = metadata.remove(ASSETS_KEY);
        String components = metadata.remove(COMPONENTS_KEY);
        ModelDescriptor restored = model;
        if (assets != null) restored = restored.withAssetCollection(decodeAssets(assets));
        if (components != null) restored = restored.withComponentCollection(decodeComponents(components));
        return restored;
    }

    static boolean hasNoGraphBindings(ComponentCollection collection) {
        return collection.components().stream().allMatch(component -> component.graphBindings().isEmpty());
    }

    static boolean isEncodedKey(String key) {
        return ASSETS_KEY.equals(key) || COMPONENTS_KEY.equals(key);
    }

    private static String encodeAssets(AssetCollection collection) {
        JsonObject root = new JsonObject();
        root.addProperty("version", 1);
        JsonArray assets = new JsonArray();
        collection.assets().forEach(asset -> {
            JsonObject item = new JsonObject();
            item.addProperty("id", asset.id());
            item.addProperty("location", asset.location());
            addOptional(item, "mediaType", asset.mediaType().orElse(null));
            item.addProperty("sizeBytes", asset.sizeBytes().isPresent()
                    ? Long.toString(asset.sizeBytes().getAsLong()) : null);
            addOptional(item, "checksum", asset.checksum().orElse(null));
            addOptional(item, "checksumAlgorithm", asset.checksumAlgorithm().orElse(null));
            item.add("extensions", stringMap(asset.extensions()));
            assets.add(item);
        });
        root.add("assets", assets);
        return GSON.toJson(root);
    }

    private static AssetCollection decodeAssets(String encoded) {
        JsonObject root = parseEnvelope(encoded, Set.of("version", "assets"));
        requireVersion(root);
        JsonArray values = requiredArray(root, "assets");
        List<AssetReference> assets = new ArrayList<>(values.size());
        values.forEach(element -> {
            JsonObject item = element.getAsJsonObject();
            requireKeys(item, Set.of("id", "location", "mediaType", "sizeBytes",
                    "checksum", "checksumAlgorithm", "extensions"));
            String size = nullableString(item, "sizeBytes");
            OptionalLong sizeBytes = size == null ? OptionalLong.empty()
                    : OptionalLong.of(Long.parseLong(size));
            assets.add(new AssetReference(
                    requiredString(item, "id"), requiredString(item, "location"),
                    Optional.ofNullable(nullableString(item, "mediaType")), sizeBytes,
                    Optional.ofNullable(nullableString(item, "checksum")),
                    Optional.ofNullable(nullableString(item, "checksumAlgorithm")),
                    decodeStringMap(requiredObject(item, "extensions"))));
        });
        return new AssetCollection(assets);
    }

    private static String encodeComponents(ComponentCollection collection) {
        JsonObject root = new JsonObject();
        root.addProperty("version", 1);
        JsonArray components = new JsonArray();
        collection.components().forEach(component -> {
            JsonObject item = new JsonObject();
            item.addProperty("id", component.id());
            item.addProperty("kind", component.kind().name());
            JsonObject configuration = new JsonObject();
            component.configuration().forEach((key, value) -> configuration.add(key, encodeValue(value)));
            item.add("configuration", configuration);
            JsonArray assetIds = new JsonArray();
            component.assetIds().forEach(assetIds::add);
            item.add("assetIds", assetIds);
            JsonArray bindings = new JsonArray();
            component.graphBindings().forEach(binding -> {
                JsonObject value = new JsonObject();
                value.addProperty("graphId", binding.graphId());
                value.add("inputPorts", stringMap(binding.inputPorts()));
                value.add("outputPorts", stringMap(binding.outputPorts()));
                bindings.add(value);
            });
            item.add("graphBindings", bindings);
            item.add("extensions", stringMap(component.extensions()));
            components.add(item);
        });
        root.add("components", components);
        return GSON.toJson(root);
    }

    private static ComponentCollection decodeComponents(String encoded) {
        JsonObject root = parseEnvelope(encoded, Set.of("version", "components"));
        requireVersion(root);
        JsonArray values = requiredArray(root, "components");
        List<ModelComponent> components = new ArrayList<>(values.size());
        values.forEach(element -> {
            JsonObject item = element.getAsJsonObject();
            requireKeys(item, Set.of("id", "kind", "configuration", "assetIds",
                    "graphBindings", "extensions"));
            Map<String, ConfigurationValue> configuration = new LinkedHashMap<>();
            requiredObject(item, "configuration").entrySet().forEach(entry ->
                    configuration.put(entry.getKey(), decodeValue(entry.getValue())));
            List<String> assetIds = stringList(requiredArray(item, "assetIds"));
            List<ComponentGraphBinding> bindings = new ArrayList<>();
            requiredArray(item, "graphBindings").forEach(bindingElement -> {
                JsonObject binding = bindingElement.getAsJsonObject();
                requireKeys(binding, Set.of("graphId", "inputPorts", "outputPorts"));
                bindings.add(new ComponentGraphBinding(requiredString(binding, "graphId"),
                        decodeStringMap(requiredObject(binding, "inputPorts")),
                        decodeStringMap(requiredObject(binding, "outputPorts"))));
            });
            components.add(new ModelComponent(requiredString(item, "id"),
                    ModelComponentKind.valueOf(requiredString(item, "kind")),
                    configuration, assetIds, bindings,
                    decodeStringMap(requiredObject(item, "extensions"))));
        });
        return new ComponentCollection(components);
    }

    private static JsonObject encodeValue(ConfigurationValue value) {
        JsonObject encoded = new JsonObject();
        if (value instanceof ConfigurationValue.StringValue v) {
            encoded.addProperty("type", "string"); encoded.addProperty("value", v.value());
        } else if (value instanceof ConfigurationValue.IntegerValue v) {
            encoded.addProperty("type", "integer"); encoded.addProperty("value", Long.toString(v.value()));
        } else if (value instanceof ConfigurationValue.DecimalValue v) {
            encoded.addProperty("type", "decimal"); encoded.addProperty("value", v.value().toPlainString());
        } else if (value instanceof ConfigurationValue.BooleanValue v) {
            encoded.addProperty("type", "boolean"); encoded.addProperty("value", v.value());
        } else if (value instanceof ConfigurationValue.StringListValue v) {
            encoded.addProperty("type", "string-list");
            JsonArray list = new JsonArray(); v.value().forEach(list::add); encoded.add("value", list);
        } else if (value instanceof ConfigurationValue.IntegerListValue v) {
            encoded.addProperty("type", "integer-list");
            JsonArray list = new JsonArray(); v.value().forEach(number -> list.add(Long.toString(number)));
            encoded.add("value", list);
        } else {
            throw new IllegalArgumentException("Unsupported AMIR component configuration value: " + value);
        }
        return encoded;
    }

    private static ConfigurationValue decodeValue(JsonElement element) {
        JsonObject value = element.getAsJsonObject();
        requireKeys(value, Set.of("type", "value"));
        return switch (requiredString(value, "type")) {
            case "string" -> new ConfigurationValue.StringValue(requiredString(value, "value"));
            case "integer" -> new ConfigurationValue.IntegerValue(Long.parseLong(requiredString(value, "value")));
            case "decimal" -> new ConfigurationValue.DecimalValue(new BigDecimal(requiredString(value, "value")));
            case "boolean" -> new ConfigurationValue.BooleanValue(requiredBoolean(value, "value"));
            case "string-list" -> new ConfigurationValue.StringListValue(stringList(requiredArray(value, "value")));
            case "integer-list" -> new ConfigurationValue.IntegerListValue(
                    requiredArray(value, "value").asList().stream()
                            .map(JsonElement::getAsString).map(Long::parseLong).toList());
            default -> throw new IllegalArgumentException("Unknown AMIR component configuration value type");
        };
    }

    private static JsonObject parseEnvelope(String encoded, Set<String> keys) {
        try {
            JsonObject root = JsonParser.parseString(encoded).getAsJsonObject();
            requireKeys(root, keys);
            return root;
        } catch (RuntimeException invalid) {
            throw new IllegalArgumentException("Invalid AMIR model layer envelope", invalid);
        }
    }

    private static void requireVersion(JsonObject root) {
        if (!root.has("version") || !root.get("version").isJsonPrimitive()
                || !root.get("version").getAsJsonPrimitive().isNumber()
                || root.get("version").getAsInt() != 1) {
            throw new IllegalArgumentException("Unsupported AMIR model layer envelope version");
        }
    }

    private static void requireKeys(JsonObject object, Set<String> keys) {
        if (!object.keySet().equals(keys)) {
            throw new IllegalArgumentException("Unexpected or missing field in AMIR model layer envelope");
        }
    }

    private static JsonObject stringMap(Map<String, String> values) {
        JsonObject result = new JsonObject();
        values.forEach(result::addProperty);
        return result;
    }

    private static Map<String, String> decodeStringMap(JsonObject object) {
        Map<String, String> result = new LinkedHashMap<>();
        object.entrySet().forEach(entry -> {
            if (!entry.getValue().isJsonPrimitive()
                    || !entry.getValue().getAsJsonPrimitive().isString()) {
                throw new IllegalArgumentException("AMIR model layer map values must be strings");
            }
            result.put(entry.getKey(), entry.getValue().getAsString());
        });
        return result;
    }

    private static List<String> stringList(JsonArray values) {
        List<String> result = new ArrayList<>(values.size());
        values.forEach(value -> {
            if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
                throw new IllegalArgumentException("Expected string in AMIR model layer list");
            }
            result.add(value.getAsString());
        });
        return result;
    }

    private static JsonArray requiredArray(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonArray()) throw new IllegalArgumentException("Missing array: " + key);
        return value.getAsJsonArray();
    }

    private static JsonObject requiredObject(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonObject()) throw new IllegalArgumentException("Missing object: " + key);
        return value.getAsJsonObject();
    }

    private static String requiredString(JsonObject object, String key) {
        String value = nullableString(object, key);
        if (value == null) throw new IllegalArgumentException("Missing string: " + key);
        return value;
    }

    private static String nullableString(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || value.isJsonNull()) return null;
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new IllegalArgumentException("Expected nullable string: " + key);
        }
        return value.getAsString();
    }

    private static boolean requiredBoolean(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean()) {
            throw new IllegalArgumentException("Expected boolean: " + key);
        }
        return value.getAsBoolean();
    }

    private static void addOptional(JsonObject object, String key, String value) {
        if (value == null) object.add(key, com.google.gson.JsonNull.INSTANCE);
        else object.addProperty(key, value);
    }
}
