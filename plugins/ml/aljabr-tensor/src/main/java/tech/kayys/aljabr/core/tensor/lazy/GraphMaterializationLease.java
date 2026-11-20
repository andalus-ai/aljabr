package tech.kayys.aljabr.core.tensor.lazy;

import tech.kayys.aljabr.core.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pins the operation-node materializations reachable from one lazy graph root.
 *
 * <p>Leases are identity-based because graph nodes form a DAG and distinct
 * nodes may have equal structure while representing different computations.
 * CONSTANT leaves remain borrowed from their caller.</p>
 *
 * <p>This type is package-private so graph lifecycle integration can evolve
 * without adding a public ownership API to Tensor.</p>
 */
final class GraphMaterializationLease implements AutoCloseable {

    private final List<GraphNode> nodesInTopologicalOrder;
    private final AtomicBoolean closed = new AtomicBoolean();

    private GraphMaterializationLease(List<GraphNode> nodesInTopologicalOrder) {
        this.nodesInTopologicalOrder = List.copyOf(nodesInTopologicalOrder);
        int acquired = 0;
        try {
            for (GraphNode node : this.nodesInTopologicalOrder) {
                node.acquireMaterializationLease();
                acquired++;
            }
        } catch (RuntimeException | Error failure) {
            for (int index = acquired - 1; index >= 0; index--) {
                try {
                    this.nodesInTopologicalOrder.get(index).releaseMaterializationLease();
                } catch (RuntimeException | Error cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
    }

    static GraphMaterializationLease acquire(GraphNode root) {
        if (root == null) {
            throw new NullPointerException("root");
        }
        List<GraphNode> ordered = new ArrayList<>();
        Set<GraphNode> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        visit(root, visited, ordered);
        return new GraphMaterializationLease(ordered);
    }

    private static void visit(GraphNode node, Set<GraphNode> visited, List<GraphNode> ordered) {
        if (!visited.add(node)) {
            return;
        }
        for (GraphNode input : node.getInputs()) {
            visit(input, visited, ordered);
        }
        if (node.getOp() != OpType.CONSTANT) {
            ordered.add(node);
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        List<Tensor> materializationsToRelease = new ArrayList<>();
        Throwable failure = null;
        for (int index = nodesInTopologicalOrder.size() - 1; index >= 0; index--) {
            try {
                Tensor materialized = nodesInTopologicalOrder.get(index).releaseMaterializationLease();
                if (materialized != null) {
                    materializationsToRelease.add(materialized);
                }
            } catch (RuntimeException | Error cleanupFailure) {
                if (failure == null) {
                    failure = cleanupFailure;
                } else {
                    failure.addSuppressed(cleanupFailure);
                }
            }
        }
        for (Tensor materialized : materializationsToRelease) {
            try {
                materialized.release();
            } catch (RuntimeException | Error cleanupFailure) {
                if (failure == null) {
                    failure = cleanupFailure;
                } else {
                    failure.addSuppressed(cleanupFailure);
                }
            }
        }
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error errorFailure) {
            throw errorFailure;
        }
    }
}
