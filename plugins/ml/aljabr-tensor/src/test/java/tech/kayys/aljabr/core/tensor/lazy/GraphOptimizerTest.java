package tech.kayys.aljabr.core.tensor.lazy;

import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.BackwardFunction;

class GraphOptimizerTest {

    private static final Shape SHAPE = new Shape(1);

    @Test
    void preservesBackwardRuleWhenAnInputRewriteRebuildsNode() {
        GraphNode leaf = node(OpType.RELU, List.of(), new Object[0], null);
        GraphNode removable = node(OpType.ADD_SCALAR, List.of(leaf), new Object[] { 0.0f }, null);
        BackwardFunction backward = (output, gradient, inputs, arguments) -> new Tensor[] { null, null };
        GraphNode root = node(OpType.ADD, List.of(removable, leaf), new Object[0], backward);

        GraphNode optimized = GraphOptimizer.optimize(root);

        assertNotSame(root, optimized);
        assertSame(leaf, optimized.getInputs().get(0));
        assertSame(backward, optimized.getBackwardFunction());
    }

    @Test
    void doesNotRemoveZeroScalarOperationWithExplicitBackwardRule() {
        GraphNode input = differentiableLeaf();
        BackwardFunction backward = (output, gradient, inputs, arguments) -> new Tensor[] { null };
        GraphNode addZero = node(OpType.ADD_SCALAR, List.of(input), new Object[] { 0.0f }, backward);

        assertSame(addZero, GraphOptimizer.optimize(addZero));
    }

    @Test
    void doesNotFuseAttentionPatternAcrossRegisteredBackwardRule() {
        GraphNode query = differentiableLeaf();
        GraphNode key = node(OpType.RELU, List.of(), new Object[0], null);
        GraphNode value = node(OpType.RELU, List.of(), new Object[0], null);
        GraphNode transpose = node(OpType.TRANSPOSE, List.of(key), new Object[0], null);
        GraphNode scores = node(OpType.MATMUL, List.of(query, transpose), new Object[0], null);
        GraphNode scaled = node(OpType.DIV_SCALAR, List.of(scores), new Object[] { 2.0f }, null);
        BackwardFunction softmaxBackward = (output, gradient, inputs, arguments) -> new Tensor[] { null };
        GraphNode probabilities = node(OpType.SOFTMAX, List.of(scaled), new Object[0], softmaxBackward);
        GraphNode root = node(OpType.MATMUL, List.of(probabilities, value), new Object[0], null);

        assertSame(root, GraphOptimizer.optimize(root));
    }

    private static GraphNode differentiableLeaf() {
        Tensor value = new MockTensor(SHAPE, DType.F32) {
            @Override
            public boolean requiresGrad() {
                return true;
            }
        };
        return GraphNode.constant(value);
    }

    private static GraphNode node(OpType op, List<GraphNode> inputs, Object[] arguments,
            BackwardFunction backward) {
        return new GraphNode(op, inputs, arguments, SHAPE, DType.F32, backward);
    }
}
