package tech.kayys.aljabr.core.tensor.lazy;

import java.util.HashMap;
import java.util.Map;

/**
 * Optimizes a computation DAG by applying fusion and rewrite rules.
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

    private static GraphNode optimizeNode(GraphNode node, Map<GraphNode, GraphNode> memo) {
        if (memo.containsKey(node)) {
            return memo.get(node);
        }

        // Optimize inputs recursively
        var newInputs = node.getInputs().stream()
                .map(input -> optimizeNode(input, memo))
                .toList();

        // Very basic optimization rule: ADD(x, 0) -> x
        if (node.getOp() == OpType.ADD_SCALAR && node.getArgs().length == 1) {
            float scalar = (float) node.getArgs()[0];
            if (scalar == 0.0f) {
                memo.put(node, newInputs.get(0));
                return newInputs.get(0);
            }
        }

        // Flash Attention Fusion Rule: MATMUL(SOFTMAX(DIV_SCALAR(MATMUL(Q,
        // TRANSPOSE(K)), scale)), V) -> ATTENTION(Q, K, V)
        if (node.getOp() == OpType.MATMUL && newInputs.size() == 2) {
            GraphNode maybeSoftmax = newInputs.get(0);
            GraphNode V = newInputs.get(1);

            if (maybeSoftmax.getOp() == OpType.SOFTMAX && maybeSoftmax.getInputs().size() == 1) {
                GraphNode maybeDiv = maybeSoftmax.getInputs().get(0);

                if ((maybeDiv.getOp() == OpType.DIV_SCALAR || maybeDiv.getOp() == OpType.MUL_SCALAR)
                        && maybeDiv.getInputs().size() == 1) {
                    GraphNode maybeMatmul = maybeDiv.getInputs().get(0);

                    if (maybeMatmul.getOp() == OpType.MATMUL && maybeMatmul.getInputs().size() == 2) {
                        GraphNode Q = maybeMatmul.getInputs().get(0);
                        GraphNode maybeTranspose = maybeMatmul.getInputs().get(1);

                        if (maybeTranspose.getOp() == OpType.TRANSPOSE && maybeTranspose.getInputs().size() == 1) {
                            GraphNode K = maybeTranspose.getInputs().get(0);

                            // We've found the full pattern! Fuse it into ATTENTION
                            GraphNode fusedNode = new GraphNode(
                                    OpType.ATTENTION,
                                    java.util.List.of(Q, K, V),
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

        // Return a new node if inputs changed.
        GraphNode optimized = new GraphNode(node.getOp(), newInputs, node.getArgs(), node.getOutputShape(),
                node.getOutputDType());
        if (node.isEvaluated()) {
            optimized.setMaterializedData(node.getMaterializedData());
        }

        memo.put(node, optimized);
        return optimized;
    }
}
