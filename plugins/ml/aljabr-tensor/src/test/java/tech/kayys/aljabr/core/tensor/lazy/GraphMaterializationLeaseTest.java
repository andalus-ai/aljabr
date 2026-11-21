package tech.kayys.aljabr.core.tensor.lazy;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GraphMaterializationLeaseTest {

    @Test
    void sharedUpstreamMaterializationSurvivesUntilLastGraphLeaseCloses() {
        AtomicInteger leafReleases = new AtomicInteger();
        AtomicInteger sharedReleases = new AtomicInteger();
        AtomicInteger leftReleases = new AtomicInteger();
        AtomicInteger rightReleases = new AtomicInteger();

        GraphNode leaf = GraphNode.constant(tensor(leafReleases));
        GraphNode shared = operation(OpType.RELU, List.of(leaf));
        GraphNode left = operation(OpType.SIGMOID, List.of(shared));
        GraphNode right = operation(OpType.TANH, List.of(shared));

        shared.setMaterializedData(tensor(sharedReleases));
        left.setMaterializedData(tensor(leftReleases));
        right.setMaterializedData(tensor(rightReleases));

        GraphMaterializationLease leftLease = GraphMaterializationLease.acquire(left);
        GraphMaterializationLease rightLease = GraphMaterializationLease.acquire(right);

        assertEquals(2, shared.materializationLeaseCount());

        leftLease.close();

        assertEquals(1, shared.materializationLeaseCount());
        assertTrue(shared.isEvaluated(), "the surviving graph still needs the shared output");
        assertEquals(0, sharedReleases.get(), "shared output must not be released prematurely");
        assertEquals(1, leftReleases.get(), "the left-only output is no longer needed");
        assertEquals(0, rightReleases.get(), "the other graph remains live");

        rightLease.close();

        assertEquals(0, shared.materializationLeaseCount());
        assertFalse(shared.isEvaluated(), "final lease clears the shared node cache");
        assertFalse(right.isEvaluated(), "final lease clears the right root cache");
        assertEquals(1, sharedReleases.get(), "shared output is released exactly once");
        assertEquals(1, rightReleases.get());
        assertEquals(0, leafReleases.get(), "CONSTANT tensors remain caller-owned");
    }

    @Test
    void closingSameLeaseRepeatedlyDoesNotReleaseMaterializationTwice() {
        AtomicInteger outputReleases = new AtomicInteger();
        GraphNode root = operation(OpType.RELU, List.of(GraphNode.constant(tensor(new AtomicInteger()))));
        root.setMaterializedData(tensor(outputReleases));

        GraphMaterializationLease lease = GraphMaterializationLease.acquire(root);
        lease.close();
        lease.close();

        assertEquals(1, outputReleases.get());
        assertFalse(root.isEvaluated());
    }

    @Test
    void releasesDependentOutputsBeforeTheirInputs() {
        StringBuilder releaseOrder = new StringBuilder();
        GraphNode leaf = GraphNode.constant(tensor(new AtomicInteger()));
        GraphNode intermediate = operation(OpType.RELU, List.of(leaf));
        GraphNode root = operation(OpType.SIGMOID, List.of(intermediate));
        intermediate.setMaterializedData(recordingTensor(releaseOrder, "intermediate"));
        root.setMaterializedData(recordingTensor(releaseOrder, "root"));

        GraphMaterializationLease.acquire(root).close();

        assertEquals("root,intermediate", releaseOrder.toString());
    }

    private static GraphNode operation(OpType op, List<GraphNode> inputs) {
        return new GraphNode(op, inputs, new Object[0], new Shape(2, 2), DType.F32);
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

    private static Tensor recordingTensor(StringBuilder order, String name) {
        Shape shape = new Shape(2, 2);
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "shape" -> shape;
                    case "dtype" -> DType.F32;
                    case "device" -> DeviceType.CPU;
                    case "requiresGrad" -> false;
                    case "release" -> {
                        if (!order.isEmpty()) {
                            order.append(',');
                        }
                        order.append(name);
                        yield null;
                    }
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    case "toString" -> name;
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }
}
