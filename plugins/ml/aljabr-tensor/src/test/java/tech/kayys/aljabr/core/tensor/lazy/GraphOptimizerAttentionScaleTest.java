package tech.kayys.aljabr.core.tensor.lazy;

import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;

class GraphOptimizerAttentionScaleTest {

    @Test
    void doesNotFuseAttentionWithCustomDivisionScale() {
        GraphNode root = attentionGraph(OpType.DIV_SCALAR, 3.0f);

        assertSame(root, GraphOptimizer.optimize(root));
    }

    @Test
    void doesNotFuseAttentionWithCustomMultiplicationScale() {
        GraphNode root = attentionGraph(OpType.MUL_SCALAR, 0.25f);

        assertSame(root, GraphOptimizer.optimize(root));
    }

    @Test
    void fusesAttentionWithStandardReciprocalMultiplicationScale() {
        GraphNode root = attentionGraph(OpType.MUL_SCALAR, 0.5f);

        GraphNode optimized = GraphOptimizer.optimize(root);

        assertNotSame(root, optimized);
        org.junit.jupiter.api.Assertions.assertSame(OpType.ATTENTION, optimized.getOp());
    }

    private static GraphNode attentionGraph(OpType scaleOperation, float scale) {
        Shape matrix = new Shape(4, 4);
        GraphNode query = node(OpType.RELU, List.of(), new Object[0], matrix);
        GraphNode key = node(OpType.RELU, List.of(), new Object[0], matrix);
        GraphNode value = node(OpType.RELU, List.of(), new Object[0], matrix);
        GraphNode transpose = node(OpType.TRANSPOSE, List.of(key), new Object[0], matrix);
        GraphNode scores = node(OpType.MATMUL, List.of(query, transpose), new Object[0], matrix);
        GraphNode scaled = node(scaleOperation, List.of(scores), new Object[] { scale }, matrix);
        GraphNode probabilities = node(OpType.SOFTMAX, List.of(scaled), new Object[0], matrix);
        return node(OpType.MATMUL, List.of(probabilities, value), new Object[0], matrix);
    }

    private static GraphNode node(OpType op, List<GraphNode> inputs, Object[] arguments, Shape shape) {
        return new GraphNode(op, inputs, arguments, shape, DType.F32);
    }
}
