package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import tech.kayys.aljabr.spi.model.ir.Dimension;
import tech.kayys.aljabr.spi.model.ir.QuantizationDescriptor;
import tech.kayys.aljabr.spi.model.ir.TensorDataType;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;
import tech.kayys.aljabr.spi.model.ir.TensorLayout;
import tech.kayys.aljabr.spi.model.ir.TensorShape;
import tech.kayys.aljabr.spi.model.ir.TensorType;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/** Preserves complete AMIR tensor types in a versioned string-metadata envelope. */
final class AmirTensorTypeCodec {
    static final String PREFIX = "aljabr.amir.tensor-type.v1.";

    private AmirTensorTypeCodec() { }

    static Map<String, String> encode(List<TensorDescriptor> tensors) {
        Map<String, String> encoded = new LinkedHashMap<>();
        for (TensorDescriptor tensor : tensors) {
            tensor.tensorType().ifPresent(type -> {
                String key = keyFor(tensor.name());
                if (encoded.putIfAbsent(key, encodeType(type)) != null) {
                    throw new IllegalArgumentException("Duplicate typed tensor metadata for " + tensor.name());
                }
            });
        }
        return Map.copyOf(encoded);
    }

    static boolean isEncodedKey(String key) {
        return key.startsWith(PREFIX);
    }

    static Map<String, String> withoutEncodedValues(Map<String, String> metadata) {
        Map<String, String> ordinary = new LinkedHashMap<>();
        metadata.forEach((key, value) -> {
            if (!isEncodedKey(key)) ordinary.put(key, value);
        });
        return Map.copyOf(ordinary);
    }

    static List<TensorDescriptor> restore(Map<String, String> metadata, List<TensorDescriptor> tensors) {
        Map<String, TensorDescriptor> byName = new LinkedHashMap<>();
        tensors.forEach(tensor -> byName.put(tensor.name(), tensor));
        Map<String, TensorType> decoded = new LinkedHashMap<>();
        metadata.forEach((key, value) -> {
            if (!isEncodedKey(key)) return;
            String name = nameFrom(key);
            if (!byName.containsKey(name)) {
                throw new IllegalArgumentException("AMIR typed tensor metadata refers to missing tensor: " + name);
            }
            try {
                if (decoded.putIfAbsent(name, decodeType(value)) != null) {
                    throw new IllegalArgumentException("Duplicate typed tensor metadata for " + name);
                }
            } catch (RuntimeException invalid) {
                throw new IllegalArgumentException("Invalid AMIR typed tensor metadata for " + name, invalid);
            }
        });
        List<TensorDescriptor> restored = new ArrayList<>(tensors.size());
        for (TensorDescriptor tensor : tensors) {
            TensorType type = decoded.get(tensor.name());
            restored.add(type == null ? tensor
                    : TensorDescriptor.typed(tensor.name(), type, tensor.payload(), tensor.extensions()));
        }
        return List.copyOf(restored);
    }

    private static String keyFor(String name) {
        return PREFIX + Base64.getUrlEncoder().withoutPadding()
                .encodeToString(name.getBytes(StandardCharsets.UTF_8));
    }

    private static String nameFrom(String key) {
        try {
            String name = new String(Base64.getUrlDecoder().decode(key.substring(PREFIX.length())),
                    StandardCharsets.UTF_8);
            if (name.isBlank()) throw new IllegalArgumentException("Tensor name must not be blank");
            return name;
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Invalid AMIR typed tensor metadata key: " + key, invalid);
        }
    }

    private static String encodeType(TensorType type) {
        JsonObject root = new JsonObject();
        root.add("dataType", encodeDataType(type.dataType()));
        JsonArray dimensions = new JsonArray();
        for (Dimension dimension : type.shape().dimensions()) {
            JsonObject item = new JsonObject();
            addOptionalLong(item, "fixed", dimension.fixedValue());
            if (dimension.symbol() == null) item.add("symbol", com.google.gson.JsonNull.INSTANCE);
            else item.addProperty("symbol", dimension.symbol());
            addOptionalLong(item, "minimum", dimension.minimum());
            addOptionalLong(item, "maximum", dimension.maximum());
            dimensions.add(item);
        }
        root.add("dimensions", dimensions);
        JsonObject layout = new JsonObject();
        layout.addProperty("id", type.layout().id());
        JsonArray strides = new JsonArray();
        type.layout().strides().forEach(strides::add);
        layout.add("strides", strides);
        root.add("layout", layout);
        if (type.quantization().isPresent()) {
            QuantizationDescriptor q = type.quantization().orElseThrow();
            JsonObject quantization = new JsonObject();
            quantization.addProperty("scheme", q.scheme());
            quantization.add("storageType", encodeDataType(q.storageType()));
            if (q.axis().isPresent()) quantization.addProperty("axis", q.axis().getAsInt());
            else quantization.add("axis", com.google.gson.JsonNull.INSTANCE);
            q.scale().ifPresent(scale -> quantization.addProperty("scale", scale.toString()));
            if (q.scale().isEmpty()) quantization.add("scale", com.google.gson.JsonNull.INSTANCE);
            addOptionalLong(quantization, "zeroPoint", q.zeroPoint());
            JsonObject extensions = new JsonObject();
            q.extensions().forEach(extensions::addProperty);
            quantization.add("extensions", extensions);
            root.add("quantization", quantization);
        } else {
            root.add("quantization", com.google.gson.JsonNull.INSTANCE);
        }
        return root.toString();
    }

    private static TensorType decodeType(String encoded) {
        JsonObject root = JsonParser.parseString(encoded).getAsJsonObject();
        TensorDataType dataType = decodeDataType(root.getAsJsonObject("dataType"));
        List<Dimension> dimensions = new ArrayList<>();
        for (JsonElement element : root.getAsJsonArray("dimensions")) {
            JsonObject item = element.getAsJsonObject();
            OptionalLong fixed = readOptionalLong(item.get("fixed"));
            String symbol = item.get("symbol").isJsonNull() ? null : item.get("symbol").getAsString();
            dimensions.add(new Dimension(fixed, symbol, readOptionalLong(item.get("minimum")),
                    readOptionalLong(item.get("maximum"))));
        }
        JsonObject layoutJson = root.getAsJsonObject("layout");
        List<Long> strides = new ArrayList<>();
        layoutJson.getAsJsonArray("strides").forEach(value -> strides.add(value.getAsLong()));
        TensorLayout layout = new TensorLayout(layoutJson.get("id").getAsString(), strides);
        Optional<QuantizationDescriptor> quantization = Optional.empty();
        JsonElement qElement = root.get("quantization");
        if (qElement != null && !qElement.isJsonNull()) {
            JsonObject q = qElement.getAsJsonObject();
            OptionalInt axis = q.get("axis").isJsonNull()
                    ? OptionalInt.empty() : OptionalInt.of(q.get("axis").getAsInt());
            Optional<BigDecimal> scale = q.get("scale").isJsonNull()
                    ? Optional.empty() : Optional.of(new BigDecimal(q.get("scale").getAsString()));
            JsonObject extensionsJson = q.getAsJsonObject("extensions");
            Map<String, String> extensions = new LinkedHashMap<>();
            extensionsJson.entrySet().forEach(entry -> extensions.put(entry.getKey(), entry.getValue().getAsString()));
            quantization = Optional.of(new QuantizationDescriptor(q.get("scheme").getAsString(),
                    decodeDataType(q.getAsJsonObject("storageType")), axis, scale,
                    readOptionalLong(q.get("zeroPoint")), extensions));
        }
        return new TensorType(dataType, new TensorShape(dimensions), layout, quantization);
    }

    private static JsonObject encodeDataType(TensorDataType type) {
        JsonObject json = new JsonObject();
        json.addProperty("id", type.id());
        json.addProperty("category", type.category().name());
        json.addProperty("bitWidth", type.bitWidth());
        return json;
    }

    private static TensorDataType decodeDataType(JsonObject json) {
        return new TensorDataType(json.get("id").getAsString(),
                TensorDataType.Category.valueOf(json.get("category").getAsString()),
                json.get("bitWidth").getAsInt());
    }

    private static void addOptionalLong(JsonObject json, String key, OptionalLong value) {
        if (value.isPresent()) json.addProperty(key, Long.toString(value.getAsLong()));
        else json.add(key, com.google.gson.JsonNull.INSTANCE);
    }

    private static OptionalLong readOptionalLong(JsonElement value) {
        return value == null || value.isJsonNull()
                ? OptionalLong.empty() : OptionalLong.of(Long.parseLong(value.getAsString()));
    }
}
