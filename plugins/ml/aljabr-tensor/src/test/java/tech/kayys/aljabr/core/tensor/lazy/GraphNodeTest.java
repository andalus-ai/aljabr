package tech.kayys.aljabr.core.tensor.lazy;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

class GraphNodeTest {

    @Test
    void preservesCompatibilityConstructorWithoutBackwardFunction() {
        GraphNode node = new GraphNode(OpType.ADD, List.of(), new Object[0], new Shape(1), DType.F32);

        assertFalse(node.hasBackwardFunction());
        assertNull(node.getBackwardFunction());
    }

    @Test
    void exposesOperationSpecificBackwardFunction() {
        var backward = (tech.kayys.aljabr.core.tensor.autograd.BackwardFunction)
                (output, outputGradient, inputs, arguments) -> new Tensor[] { null, null };
        GraphNode node = new GraphNode(OpType.ADD, List.of(), new Object[0], new Shape(1), DType.F32, backward);

        assertTrue(node.hasBackwardFunction());
        assertSame(backward, node.getBackwardFunction());
    }
}
