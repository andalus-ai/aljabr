package tech.kayys.aljabr.core.tensor.lazy;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class LazyTensorGraphLeaseIntegrationTest {

    @Test
    void childGraphKeepsSharedParentMaterializationAlive() {
        AtomicInteger parentReleases = new AtomicInteger();
        AtomicInteger childReleases = new AtomicInteger();
        GraphNode parentNode = operation(OpType.RELU, List.of(GraphNode.constant(tensor(new AtomicInteger()))));
        GraphNode childNode = operation(OpType.SIGMOID, List.of(parentNode));
        parentNode.setMaterializedData(tensor(parentReleases));
        childNode.setMaterializedData(tensor(childReleases));

        LazyTensor parent = new LazyTensor(parentNode, null);
        LazyTensor child = new LazyTensor(childNode, null);

        parent.release();

        assertTrue(parentNode.isEvaluated(), "the child lease must retain its shared parent output");
        assertEquals(0, parentReleases.get());

        child.release();

        assertFalse(parentNode.isEvaluated());
        assertFalse(childNode.isEvaluated());
        assertEquals(1, parentReleases.get());
        assertEquals(1, childReleases.get());
    }

    @Test
    void optimizedIdentityRewriteDoesNotTransferOrDoubleReleaseParentOutput() {
        AtomicInteger releases = new AtomicInteger();
        GraphNode input = operation(OpType.RELU, List.of(GraphNode.constant(tensor(new AtomicInteger()))));
        Tensor materialized = tensor(releases);
        input.setMaterializedData(materialized);
        GraphNode addZero = operation(OpType.ADD_SCALAR, List.of(input), new Object[] { 0.0f });
        LazyTensor lazy = new LazyTensor(addZero, null);

        assertSame(materialized, lazy.eval());
        assertFalse(addZero.isEvaluated(), "the optimized root must not alias its input cache");

        lazy.release();

        assertFalse(input.isEvaluated());
        assertEquals(1, releases.get());
    }

    @Test
    void rebuiltOptimizerNodeDoesNotInheritSourceMaterialization() {
        GraphNode leaf = GraphNode.constant(tensor(new AtomicInteger()));
        GraphNode removable = operation(OpType.ADD_SCALAR, List.of(leaf), new Object[] { 0.0f });
        GraphNode root = operation(OpType.RELU, List.of(removable));
        root.setMaterializedData(tensor(new AtomicInteger()));

        GraphNode optimized = GraphOptimizer.optimize(root);

        assertFalse(optimized == root);
        assertFalse(optimized.isEvaluated(), "rewritten nodes must not borrow source cache ownership");
        assertTrue(root.isEvaluated(), "the original graph retains its own cached output");
        assertSame(leaf, optimized.getInputs().get(0));
    }

    private static GraphNode operation(OpType op, List<GraphNode> inputs) {
        return operation(op, inputs, new Object[0]);
    }

    private static GraphNode operation(OpType op, List<GraphNode> inputs, Object[] args) {
        return new GraphNode(op, inputs, args, new Shape(2, 2), DType.F32);
    }

    private static Tensor tensor(AtomicInteger releases) {
        Shape shape = new Shape(2, 2);
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "shape" -> shape;
                    case "dtype" -> DType.F32;
                    case "device" -> DeviceType.CPU;
                    case "requiresGrad" -> false;
                    case "backend" -> null;
                    case "release" -> {
                        releases.incrementAndGet();
                        yield null;
                    }
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    case "toString" -> "LeaseTrackingTensor";
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }
}
