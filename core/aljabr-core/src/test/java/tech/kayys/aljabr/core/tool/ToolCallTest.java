package tech.kayys.aljabr.core.tool;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolCallTest {
    @Test
    void deeplySnapshotsArguments() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("items", new ArrayList<>(List.of("first")));
        ToolCall call = new ToolCall("search", Map.of("filter", nested));
        ((List<String>) nested.get("items")).add("later");
        assertEquals(List.of("first"), ((Map<?, ?>) call.arguments().get("filter")).get("items"));
        assertThrows(UnsupportedOperationException.class,
                () -> ((List<Object>) ((Map<?, ?>) call.arguments().get("filter")).get("items")).add("no"));
    }

    @Test
    void rejectsUnsupportedValuesAndBlankNames() {
        assertThrows(IllegalArgumentException.class, () -> new ToolCall(" ", Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new ToolCall("x", Map.of("bad", new Object())));
    }
}
