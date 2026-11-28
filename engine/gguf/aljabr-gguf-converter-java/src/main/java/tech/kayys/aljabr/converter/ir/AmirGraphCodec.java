package tech.kayys.aljabr.converter.ir;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import tech.kayys.aljabr.spi.model.ir.ControlFlowRegion;
import tech.kayys.aljabr.spi.model.ir.GraphCollection;
import tech.kayys.aljabr.spi.model.ir.GraphDescriptor;
import tech.kayys.aljabr.spi.model.ir.GraphSignature;
import tech.kayys.aljabr.spi.model.ir.GraphValue;
import tech.kayys.aljabr.spi.model.ir.OperationDescriptor;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Versioned format-neutral serialization for AMIR computation graph collections. */
final class AmirGraphCodec {
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();
    private AmirGraphCodec() { }

    static String encode(GraphCollection collection) {
        JsonObject root = new JsonObject();
        root.addProperty("version", 1);
        JsonArray graphs = new JsonArray();
        collection.graphs().forEach(graph -> graphs.add(encodeGraph(graph)));
        root.add("graphs", graphs);
        return GSON.toJson(root);
    }

    static GraphCollection decode(String encoded) {
        JsonObject root = parse(encoded);
        requireKeys(root, Set.of("version", "graphs"));
        requireVersion(root);
        JsonArray values = array(root, "graphs");
        List<GraphDescriptor> graphs = new ArrayList<>(values.size());
        values.forEach(value -> graphs.add(decodeGraph(value.getAsJsonObject())));
        return new GraphCollection(graphs);
    }

    private static JsonObject encodeGraph(GraphDescriptor graph) {
        JsonObject result = new JsonObject();
        result.addProperty("id", graph.id());
        JsonArray values = new JsonArray();
        graph.values().forEach(value -> {
            JsonObject item = new JsonObject();
            item.addProperty("id", value.id());
            item.add("type", JsonParser.parseString(AmirTensorTypeCodec.encodeType(value.type())));
            item.addProperty("description", value.description());
            values.add(item);
        });
        result.add("values", values);
        JsonArray operations = new JsonArray();
        graph.operations().forEach(operation -> {
            JsonObject item = new JsonObject();
            item.addProperty("id", operation.id());
            item.addProperty("operation", operation.operation());
            JsonObject inputs = new JsonObject();
            operation.inputs().forEach((port, ids) -> {
                JsonArray list = new JsonArray();
                ids.forEach(list::add);
                inputs.add(port, list);
            });
            item.add("inputs", inputs);
            item.add("outputs", AmirModelLayerCodec.stringMap(operation.outputs()));
            JsonObject attributes = new JsonObject();
            operation.attributes().forEach((key, value) ->
                    attributes.add(key, AmirModelLayerCodec.encodeValue(value)));
            item.add("attributes", attributes);
            item.add("extensions", AmirModelLayerCodec.stringMap(operation.extensions()));
            operations.add(item);
        });
        result.add("operations", operations);
        JsonObject signature = new JsonObject();
        signature.add("inputs", AmirModelLayerCodec.stringMap(graph.signature().inputs()));
        signature.add("outputs", AmirModelLayerCodec.stringMap(graph.signature().outputs()));
        result.add("signature", signature);
        JsonArray regions = new JsonArray();
        graph.controlFlowRegions().forEach(region -> {
            JsonObject item = new JsonObject();
            item.addProperty("id", region.id());
            item.addProperty("kind", region.kind().name());
            item.addProperty("parentRegionId", region.parentRegionId().orElse(null));
            item.add("nodeIds", strings(region.nodeIds()));
            item.add("inputValueIds", strings(region.inputValueIds()));
            item.add("outputValueIds", strings(region.outputValueIds()));
            item.add("extensions", AmirModelLayerCodec.stringMap(region.extensions()));
            regions.add(item);
        });
        result.add("controlFlowRegions", regions);
        result.add("extensions", AmirModelLayerCodec.stringMap(graph.extensions()));
        return result;
    }

    private static GraphDescriptor decodeGraph(JsonObject item) {
        requireKeys(item, Set.of("id", "values", "operations", "signature", "controlFlowRegions", "extensions"));
        List<GraphValue> values = new ArrayList<>();
        array(item, "values").forEach(element -> {
            JsonObject value = element.getAsJsonObject();
            requireKeys(value, Set.of("id", "type", "description"));
            values.add(new GraphValue(string(value, "id"),
                    AmirTensorTypeCodec.decodeType(value.get("type").toString()), string(value, "description")));
        });
        List<OperationDescriptor> operations = new ArrayList<>();
        array(item, "operations").forEach(element -> {
            JsonObject operation = element.getAsJsonObject();
            requireKeys(operation, Set.of("id", "operation", "inputs", "outputs", "attributes", "extensions"));
            Map<String, List<String>> inputs = new LinkedHashMap<>();
            object(operation, "inputs").entrySet().forEach(entry -> {
                if (!entry.getValue().isJsonArray()) throw invalid("operation inputs must be arrays");
                inputs.put(entry.getKey(), stringList(entry.getValue().getAsJsonArray()));
            });
            Map<String, ConfigurationValue> attributes = new LinkedHashMap<>();
            object(operation, "attributes").entrySet().forEach(entry ->
                    attributes.put(entry.getKey(), AmirModelLayerCodec.decodeValue(entry.getValue())));
            operations.add(new OperationDescriptor(string(operation, "id"), string(operation, "operation"),
                    inputs, AmirModelLayerCodec.decodeStringMap(object(operation, "outputs")),
                    attributes, AmirModelLayerCodec.decodeStringMap(object(operation, "extensions"))));
        });
        JsonObject signature = object(item, "signature");
        requireKeys(signature, Set.of("inputs", "outputs"));
        GraphSignature graphSignature = new GraphSignature(
                AmirModelLayerCodec.decodeStringMap(object(signature, "inputs")),
                AmirModelLayerCodec.decodeStringMap(object(signature, "outputs")));
        List<ControlFlowRegion> regions = new ArrayList<>();
        array(item, "controlFlowRegions").forEach(element -> {
            JsonObject region = element.getAsJsonObject();
            requireKeys(region, Set.of("id", "kind", "parentRegionId", "nodeIds",
                    "inputValueIds", "outputValueIds", "extensions"));
            JsonElement parent = region.get("parentRegionId");
            Optional<String> parentId = parent == null || parent.isJsonNull()
                    ? Optional.empty() : Optional.of(parent.getAsString());
            regions.add(new ControlFlowRegion(string(region, "id"),
                    ControlFlowRegion.Kind.valueOf(string(region, "kind")), parentId,
                    stringList(array(region, "nodeIds")), stringList(array(region, "inputValueIds")),
                    stringList(array(region, "outputValueIds")),
                    AmirModelLayerCodec.decodeStringMap(object(region, "extensions"))));
        });
        return new GraphDescriptor(string(item, "id"), values, operations, graphSignature,
                regions, AmirModelLayerCodec.decodeStringMap(object(item, "extensions")));
    }

    private static JsonObject parse(String encoded) {
        try { return JsonParser.parseString(encoded).getAsJsonObject(); }
        catch (RuntimeException invalid) { throw new IllegalArgumentException("Invalid AMIR graph envelope", invalid); }
    }
    private static void requireVersion(JsonObject root) {
        JsonElement version = root.get("version");
        if (version == null || !version.isJsonPrimitive() || !version.getAsJsonPrimitive().isNumber()
                || !"1".equals(version.getAsString())) throw invalid("Unsupported AMIR graph envelope version");
    }
    private static void requireKeys(JsonObject object, Set<String> keys) {
        if (!object.keySet().equals(keys)) throw invalid("Unexpected or missing AMIR graph envelope field");
    }
    private static JsonObject object(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonObject()) throw invalid("Expected object: " + key);
        return value.getAsJsonObject();
    }
    private static JsonArray array(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonArray()) throw invalid("Expected array: " + key);
        return value.getAsJsonArray();
    }
    private static String string(JsonObject object, String key) {
        JsonElement value = object.get(key);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString())
            throw invalid("Expected string: " + key);
        return value.getAsString();
    }
    private static JsonArray strings(List<String> values) {
        JsonArray result = new JsonArray();
        values.forEach(result::add);
        return result;
    }
    private static List<String> stringList(JsonArray values) {
        List<String> result = new ArrayList<>(values.size());
        values.forEach(value -> {
            if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString())
                throw invalid("Expected string in AMIR graph list");
            result.add(value.getAsString());
        });
        return List.copyOf(result);
    }
    private static IllegalArgumentException invalid(String message) { return new IllegalArgumentException(message); }
}
