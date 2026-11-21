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


    @Test
    void doesNotFuseAttentionWhenSoftmaxUsesNonKeyAxis() {
        GraphNode root = attentionGraph(OpType.DIV_SCALAR, 2.0f, new Object[] { 0 }, new Object[0],
                new Shape(4, 4));

        assertSame(root, GraphOptimizer.optimize(root));
    }

    @Test
    void doesNotFuseAttentionWhenTransposeDoesNotSwapLastTwoAxes() {
        Shape batched = new Shape(2, 3, 4, 4);
        GraphNode root = attentionGraph(OpType.DIV_SCALAR, 2.0f, new Object[0],
                new Object[] { 0, 1 }, batched);

        assertSame(root, GraphOptimizer.optimize(root));
    }

    @Test
    void keepsSafeChildRewritesWhenCustomScalePreventsAttentionFusion() {
        Shape shape = new Shape(4, 4);
        GraphNode queryLeaf = node(OpType.RELU, List.of(), new Object[0], shape);
        GraphNode query = node(OpType.ADD_SCALAR, List.of(queryLeaf), new Object[] { 0.0f }, shape);
        GraphNode key = node(OpType.RELU, List.of(), new Object[0], shape);
        GraphNode value = node(OpType.RELU, List.of(), new Object[0], shape);
        GraphNode transpose = node(OpType.TRANSPOSE, List.of(key), new Object[0], shape);
        GraphNode scores = node(OpType.MATMUL, List.of(query, transpose), new Object[0], shape);
        GraphNode scaled = node(OpType.DIV_SCALAR, List.of(scores), new Object[] { 3.0f }, shape);
        GraphNode probabilities = node(OpType.SOFTMAX, List.of(scaled), new Object[0], shape);
        GraphNode root = node(OpType.MATMUL, List.of(probabilities, value), new Object[0], shape);

        GraphNode optimized = GraphOptimizer.optimize(root);

        assertNotSame(root, optimized);
        GraphNode optimizedQuery = optimized.getInputs().get(0).getInputs().get(0)
                .getInputs().get(0).getInputs().get(0);
        assertSame(queryLeaf, optimizedQuery);
    }

    @Test
    void doesNotFuseRankTwoAttentionWhenKeyAndValueSequenceLengthsDiffer() {
        GraphNode query = node(OpType.RELU, List.of(), new Object[0], new Shape(2, 4));
        GraphNode key = node(OpType.RELU, List.of(), new Object[0], new Shape(3, 4));
        GraphNode value = node(OpType.RELU, List.of(), new Object[0], new Shape(3, 4));
        GraphNode transpose = node(OpType.TRANSPOSE, List.of(key), new Object[0], new Shape(4, 3));
        GraphNode scores = node(OpType.MATMUL, List.of(query, transpose), new Object[0], new Shape(2, 3));
        GraphNode scaled = node(OpType.DIV_SCALAR, List.of(scores), new Object[] { 2.0f }, new Shape(2, 3));
        GraphNode probabilities = node(OpType.SOFTMAX, List.of(scaled), new Object[0], new Shape(2, 3));
        GraphNode root = node(OpType.MATMUL, List.of(probabilities, value), new Object[0], new Shape(2, 4));

        assertSame(root, GraphOptimizer.optimize(root));
    }

    @Test
    void doesNotFuseRankFourAttentionUntilBackendLayoutIsStandardized() {
        Shape queryShape = new Shape(2, 3, 4, 8);
        Shape transposeShape = new Shape(2, 3, 8, 4);
        Shape scoresShape = new Shape(2, 3, 4, 4);
        GraphNode query = node(OpType.RELU, List.of(), new Object[0], queryShape);
        GraphNode key = node(OpType.RELU, List.of(), new Object[0], queryShape);
        GraphNode value = node(OpType.RELU, List.of(), new Object[0], queryShape);
        GraphNode transpose = node(OpType.TRANSPOSE, List.of(key), new Object[] { 2, 3 }, transposeShape);
        GraphNode scores = node(OpType.MATMUL, List.of(query, transpose), new Object[0], scoresShape);
        GraphNode scaled = node(OpType.DIV_SCALAR, List.of(scores), new Object[] { (float) Math.sqrt(8) },
                scoresShape);
        GraphNode probabilities = node(OpType.SOFTMAX, List.of(scaled), new Object[0], scoresShape);
        GraphNode root = node(OpType.MATMUL, List.of(probabilities, value), new Object[0], queryShape);

        assertSame(root, GraphOptimizer.optimize(root));
    }

    private static GraphNode attentionGraph(OpType scaleOperation, float scale) {
        return attentionGraph(scaleOperation, scale, new Object[0], new Object[0], new Shape(4, 4));
    }

    private static GraphNode attentionGraph(OpType scaleOperation, float scale, Object[] softmaxArguments,
            Object[] transposeArguments, Shape matrix) {
        GraphNode query = node(OpType.RELU, List.of(), new Object[0], matrix);
        GraphNode key = node(OpType.RELU, List.of(), new Object[0], matrix);
        GraphNode value = node(OpType.RELU, List.of(), new Object[0], matrix);
        GraphNode transpose = node(OpType.TRANSPOSE, List.of(key), transposeArguments, matrix);
        GraphNode scores = node(OpType.MATMUL, List.of(query, transpose), new Object[0], matrix);
        GraphNode scaled = node(scaleOperation, List.of(scores), new Object[] { scale }, matrix);
        GraphNode probabilities = node(OpType.SOFTMAX, List.of(scaled), softmaxArguments, matrix);
        return node(OpType.MATMUL, List.of(probabilities, value), new Object[0], matrix);
    }

    private static GraphNode node(OpType op, List<GraphNode> inputs, Object[] arguments, Shape shape) {
        return new GraphNode(op, inputs, arguments, shape, DType.F32);
    }
}
