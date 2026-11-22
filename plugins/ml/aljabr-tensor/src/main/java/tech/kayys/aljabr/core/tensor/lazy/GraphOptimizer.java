package tech.kayys.aljabr.core.tensor.lazy;

import java.util.HashMap;
import java.util.Map;

import tech.kayys.aljabr.core.tensor.DType;

/**
 * Optimizes a computation DAG by applying fusion and rewrite rules.
 *
 * <p>Graph rewrites must preserve the local derivative metadata of every
 * operation they retain. Rewrites that remove or fuse nodes are only valid
 * when they do not remove a path needed by a differentiable graph.</p>
 *
 * @author bhangun
 */
public class GraphOptimizer {

    /**
     * Optimizes the graph starting from the given root node.
     *
     * @param root the root node of the computation graph
     * @return an optimized GraphNode
     */
    public static GraphNode optimize(GraphNode root) {
        Map<GraphNode, GraphNode> visited = new HashMap<>();
        return optimizeNode(root, visited);
    }

    private static boolean usesStandardAttentionScale(GraphNode scale, GraphNode query) {
        Object[] arguments = scale.getArgs();
        if (arguments.length != 1 || !(arguments[0] instanceof Number number)) {
            return false;
        }

        int rank = query.getOutputShape().rank();
        if (rank == 0) {
            return false;
        }
        long headDimension = query.getOutputShape().dim(rank - 1);
        if (headDimension <= 0) {
            return false;
        }

        float actual = number.floatValue();
        float expected = scale.getOp() == OpType.DIV_SCALAR
                ? (float) Math.sqrt(headDimension)
                : (float) (1.0d / Math.sqrt(headDimension));
        if (!Float.isFinite(actual)) {
            return false;
        }

        // Allow normal float rounding while rejecting meaningful scale changes.
        float tolerance = Math.max(1.0e-6f, Math.ulp(expected) * 4.0f);
        return Math.abs(actual - expected) <= tolerance;
    }

    private static boolean usesStandardAttentionSoftmax(GraphNode softmax) {
        Object[] arguments = softmax.getArgs();
        if (arguments.length == 0) {
            return true;
        }
        if (arguments.length != 1 || !(arguments[0] instanceof Number number)) {
            return false;
        }
        int rank = softmax.getOutputShape().rank();
        if (rank == 0) {
            return false;
        }
        int dimension = number.intValue();
        int normalized = dimension < 0 ? dimension + rank : dimension;
        return normalized == rank - 1;
    }

    /**
     * The current backend contract does not define one common rank-4 attention
     * layout: the CPU reference uses [B,H,S,D], while the Metal path consumes
     * [B,S,H,D]. Keep this graph rewrite to the unambiguous rank-2 case until
     * the backend attention layout contract is standardized.
     *
     * <p>The CPU rank-2 kernel also requires Q, K, and V to have matching
     * [sequence, headDimension] shapes. The unfused matmul chain supports
     * different key/value sequence lengths, so those graphs must remain intact.</p>
     */
    private static boolean supportsCanonicalAttentionShape(
            GraphNode output,
            GraphNode scores,
            GraphNode softmax,
            GraphNode transpose,
            GraphNode query,
            GraphNode key,
            GraphNode value) {
        long[] queryShape = query.getOutputShape().dims();
        if (queryShape.length != 2 || queryShape[0] <= 0 || queryShape[1] <= 0) {
            return false;
        }
        if (!java.util.Arrays.equals(queryShape, key.getOutputShape().dims())
                || !java.util.Arrays.equals(queryShape, value.getOutputShape().dims())) {
            return false;
        }
        long[] expectedScores = { queryShape[0], queryShape[0] };
        long[] expectedTranspose = { queryShape[1], queryShape[0] };
        return java.util.Arrays.equals(expectedScores, scores.getOutputShape().dims())
                && java.util.Arrays.equals(expectedScores, softmax.getOutputShape().dims())
                && java.util.Arrays.equals(expectedTranspose, transpose.getOutputShape().dims())
                && java.util.Arrays.equals(queryShape, output.getOutputShape().dims());
    }

    private static boolean hasConsistentAttentionDType(
            GraphNode output,
            GraphNode softmax,
            GraphNode scale,
            GraphNode scores,
            GraphNode transpose,
            GraphNode query,
            GraphNode key,
            GraphNode value) {
        var dtype = query.getOutputDType();
        // The backend-neutral ATTENTION contract currently guarantees F32 only.
        // Do not replace an unfused graph with an operation some backends reject.
        return dtype == DType.F32
                && dtype == key.getOutputDType()
                && dtype == value.getOutputDType()
                && dtype == transpose.getOutputDType()
                && dtype == scores.getOutputDType()
                && dtype == scale.getOutputDType()
                && dtype == softmax.getOutputDType()
                && dtype == output.getOutputDType();
    }

    private static boolean usesStandardAttentionTranspose(GraphNode transpose, GraphNode key) {
        Object[] arguments = transpose.getArgs();
        if (arguments.length == 0) {
            return key.getOutputShape().rank() >= 2;
        }
        if (arguments.length != 2
                || !(arguments[0] instanceof Number first)
                || !(arguments[1] instanceof Number second)) {
            return false;
        }
        int rank = key.getOutputShape().rank();
        if (rank < 2) {
            return false;
        }
        int firstDimension = first.intValue();
        int secondDimension = second.intValue();
        int normalizedFirst = firstDimension < 0 ? firstDimension + rank : firstDimension;
        int normalizedSecond = secondDimension < 0 ? secondDimension + rank : secondDimension;
        return normalizedFirst == rank - 2 && normalizedSecond == rank - 1
                || normalizedFirst == rank - 1 && normalizedSecond == rank - 2;
    }

    private static boolean isIntegralDType(DType dtype) {
        return dtype == DType.I32 || dtype == DType.I8 || dtype == DType.INT8;
    }

    private static GraphNode optimizeNode(GraphNode node, Map<GraphNode, GraphNode> memo) {
        if (memo.containsKey(node)) {
            return memo.get(node);
        }

        // Optimize inputs recursively.
        var newInputs = node.getInputs().stream()
                .map(input -> optimizeNode(input, memo))
                .toList();

        // Integer ADD_SCALAR(x, 0) -> x is safe only when no gradient path is present.
        // Floating-point addition is retained because IEEE-754 signed-zero and NaN semantics matter.
        if (node.getOp() == OpType.ADD_SCALAR
                && !node.requiresGrad()
                && newInputs.size() == 1
                && newInputs.get(0).getOutputDType() == node.getOutputDType()
                && java.util.Arrays.equals(
                        newInputs.get(0).getOutputShape().dims(), node.getOutputShape().dims())
                && isIntegralDType(node.getOutputDType())
                && node.getArgs().length == 1
                && node.getArgs()[0] instanceof Number number
                && number.doubleValue() == 0.0d) {
            memo.put(node, newInputs.get(0));
            return newInputs.get(0);
        }

        // Flash Attention Fusion:
        // MATMUL(SOFTMAX(DIV_SCALAR(MATMUL(Q, TRANSPOSE(K)), scale)), V)
        // -> ATTENTION(Q, K, V)
        //
        // Only fuse non-differentiable graphs. The fused ATTENTION node does not
        // currently provide a composite VJP equivalent to the removed nodes.
        if (node.getOp() == OpType.MATMUL && newInputs.size() == 2 && !node.requiresGrad()) {
            GraphNode maybeSoftmax = newInputs.get(0);
            GraphNode value = newInputs.get(1);

            if (maybeSoftmax.getOp() == OpType.SOFTMAX
                    && !maybeSoftmax.requiresGrad()
                    && maybeSoftmax.getInputs().size() == 1) {
                GraphNode maybeDiv = maybeSoftmax.getInputs().get(0);

                if ((maybeDiv.getOp() == OpType.DIV_SCALAR || maybeDiv.getOp() == OpType.MUL_SCALAR)
                        && !maybeDiv.requiresGrad()
                        && maybeDiv.getInputs().size() == 1) {
                    GraphNode maybeMatmul = maybeDiv.getInputs().get(0);

                    if (maybeMatmul.getOp() == OpType.MATMUL
                            && !maybeMatmul.requiresGrad()
                            && maybeMatmul.getInputs().size() == 2) {
                        GraphNode query = maybeMatmul.getInputs().get(0);
                        GraphNode maybeTranspose = maybeMatmul.getInputs().get(1);

                        if (maybeTranspose.getOp() == OpType.TRANSPOSE
                                && !maybeTranspose.requiresGrad()
                                && maybeTranspose.getInputs().size() == 1) {
                            GraphNode key = maybeTranspose.getInputs().get(0);
                            boolean standardPattern = usesStandardAttentionScale(maybeDiv, query)
                                    && usesStandardAttentionSoftmax(maybeSoftmax)
                                    && usesStandardAttentionTranspose(maybeTranspose, key)
                                    && supportsCanonicalAttentionShape(node, maybeMatmul, maybeSoftmax,
                                            maybeTranspose, query, key, value)
                                    && hasConsistentAttentionDType(node, maybeSoftmax, maybeDiv, maybeMatmul,
                                            maybeTranspose, query, key, value);
                            if (standardPattern) {
                                GraphNode fusedNode = new GraphNode(
                                        OpType.ATTENTION,
                                        java.util.List.of(query, key, value),
                                        new Object[0],
                                        node.getOutputShape(),
                                        node.getOutputDType());
                                memo.put(node, fusedNode);
                                return fusedNode;
                            }
                        }
                    }
                }
            }
        }

        boolean inputsChanged = false;
        for (int i = 0; i < node.getInputs().size(); i++) {
            if (node.getInputs().get(i) != newInputs.get(i)) {
                inputsChanged = true;
                break;
            }
        }

        if (!inputsChanged) {
            memo.put(node, node);
            return node;
        }

        // Preserve the operation's local derivative when rebuilding the node.
        GraphNode optimized = new GraphNode(
                node.getOp(),
                newInputs,
                node.getArgs(),
                node.getOutputShape(),
                node.getOutputDType(),
                node.getBackwardFunction());
        // Rewritten nodes must not inherit cached Tensor references from the
        // source graph. The source and rewritten graph have independent
        // materialization leases; sharing this pointer would double-release it.
        memo.put(node, optimized);
        return optimized;
    }
}
