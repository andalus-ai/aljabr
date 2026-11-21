package tech.kayys.aljabr.spi.model.ir;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class GraphDescriptorTest {

    private static GraphValue value(String id) {
        return new GraphValue(id, TensorType.of(TensorDataType.F32, TensorShape.of(2, 2)));
    }

    @Test
    void buildsTypedGraphWithNamedSignature() {
        OperationDescriptor matmul = new OperationDescriptor(
                "matmul-0", "linear.matmul",
                Map.of("lhs", List.of("input"), "rhs", List.of("weight")),
                Map.of("result", "output"),
                Map.of("transpose_rhs", new ConfigurationValue.BooleanValue(false)),
                Map.of("vendor", "test"));
        GraphDescriptor graph = GraphDescriptor.of(
                "forward",
                List.of(value("input"), value("weight"), value("output")),
                List.of(matmul),
                new GraphSignature(Map.of("tokens", "input"), Map.of("logits", "output")));

        assertEquals("forward", graph.id());
        assertEquals("input", graph.signature().inputs().get("tokens"));
        assertEquals("output", graph.signature().outputs().get("logits"));
        assertEquals("linear.matmul", graph.operations().getFirst().operation());
    }

    @Test
    void validatesOperationReferencesAndUniqueProducers() {
        OperationDescriptor unknownInput = new OperationDescriptor(
                "op", "custom.op", Map.of("x", List.of("missing")),
                Map.of("y", "output"), Map.of(), Map.of());
        assertThrows(IllegalArgumentException.class, () -> GraphDescriptor.of(
                "graph", List.of(value("output")), List.of(unknownInput), GraphSignature.empty()));

        OperationDescriptor first = new OperationDescriptor(
                "first", "custom.first", Map.of(), Map.of("y", "shared"), Map.of(), Map.of());
        OperationDescriptor second = new OperationDescriptor(
                "second", "custom.second", Map.of(), Map.of("y", "shared"), Map.of(), Map.of());
        assertThrows(IllegalArgumentException.class, () -> GraphDescriptor.of(
                "graph", List.of(value("shared")), List.of(first, second), GraphSignature.empty()));
    }

    @Test
    void permitsCyclesAndRepresentsControlFlowExplicitly() {
        OperationDescriptor loop = new OperationDescriptor(
                "loop-body", "custom.loop-carried",
                Map.of("previous", List.of("state"), "next", List.of("updated")),
                Map.of("updated", "updated"), Map.of(), Map.of());
        ControlFlowRegion region = new ControlFlowRegion(
                "while-region", ControlFlowRegion.Kind.LOOP, Optional.empty(),
                List.of("loop-body"), List.of("state"), List.of("updated"), Map.of());
        GraphDescriptor graph = new GraphDescriptor(
                "recurrent", List.of(value("state"), value("updated")), List.of(loop),
                new GraphSignature(Map.of("initial_state", "state"), Map.of("final_state", "updated")),
                List.of(region), Map.of());

        assertEquals(ControlFlowRegion.Kind.LOOP, graph.controlFlowRegions().getFirst().kind());
    }

    @Test
    void validatesRegionMembershipAndParentReferences() {
        ControlFlowRegion orphan = new ControlFlowRegion(
                "child", ControlFlowRegion.Kind.CONDITIONAL, Optional.of("missing"),
                List.of(), List.of(), List.of(), Map.of());
        assertThrows(IllegalArgumentException.class, () -> new GraphDescriptor(
                "graph", List.of(), List.of(), GraphSignature.empty(), List.of(orphan), Map.of()));

        ControlFlowRegion unknownNode = new ControlFlowRegion(
                "region", ControlFlowRegion.Kind.LOOP, Optional.empty(),
                List.of("missing-op"), List.of(), List.of(), Map.of());
        assertThrows(IllegalArgumentException.class, () -> new GraphDescriptor(
                "graph", List.of(), List.of(), GraphSignature.empty(), List.of(unknownNode), Map.of()));
    }

    @Test
    void validatesSignatureAndGraphCollectionIdentity() {
        assertThrows(IllegalArgumentException.class, () -> GraphDescriptor.of(
                "graph", List.of(), List.of(),
                new GraphSignature(Map.of("input", "unknown"), Map.of())));

        GraphDescriptor first = GraphDescriptor.of("same", List.of(), List.of(), GraphSignature.empty());
        GraphDescriptor second = GraphDescriptor.of("same", List.of(), List.of(), GraphSignature.empty());
        assertThrows(IllegalArgumentException.class, () -> new GraphCollection(List.of(first, second)));
    }

    @Test
    void attachesGraphsToModelDescriptorWithoutBreakingConstructors() {
        ModelDescriptor model = new ModelDescriptor(
                ModelIdentity.of("sample"), "decoder", "native", Map.of(), Map.of(), List.of());
        GraphDescriptor graph = GraphDescriptor.of("forward", List.of(), List.of(), GraphSignature.empty());

        ModelDescriptor enriched = model.withGraphCollection(new GraphCollection(List.of(graph)));
        assertTrue(model.graphCollection().isEmpty());
        assertEquals("forward", enriched.graphCollection().orElseThrow().graphs().getFirst().id());
        assertEquals(model, model.withIdentity(ModelIdentity.of("renamed"))
                .withIdentity(model.identity()));
    }
}
