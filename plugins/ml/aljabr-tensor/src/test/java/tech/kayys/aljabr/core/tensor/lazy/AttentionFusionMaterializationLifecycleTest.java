package tech.kayys.aljabr.core.tensor.lazy;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;

class AttentionFusionMaterializationLifecycleTest {

    @Test
    void fusedAttentionLeaseReleasesOutputAndSharedInputsExactlyOnce() {
        AtomicInteger leafReleases = new AtomicInteger();
        AtomicInteger queryReleases = new AtomicInteger();
        AtomicInteger keyReleases = new AtomicInteger();
        AtomicInteger valueReleases = new AtomicInteger();
        AtomicInteger outputReleases = new AtomicInteger();

        GraphNode query = materializedOperation(tensor(leafReleases), queryReleases);
        GraphNode key = materializedOperation(tensor(leafReleases), keyReleases);
        GraphNode value = materializedOperation(tensor(leafReleases), valueReleases);

        GraphNode transpose = operation(OpType.TRANSPOSE, List.of(key));
        GraphNode scores = operation(OpType.MATMUL, List.of(query, transpose));
        GraphNode scaled = operation(OpType.DIV_SCALAR, List.of(scores), new Object[] { 2.0f });
        GraphNode probabilities = operation(OpType.SOFTMAX, List.of(scaled));
        GraphNode root = operation(OpType.MATMUL, List.of(probabilities, value));

        Tensor output = tensor(outputReleases);
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("attention")) {
                        return output;
                    }
                    if (method.getName().equals("toString")) {
                        return "AttentionTrackingBackend";
                    }
                    if (method.getName().equals("hashCode")) {
                        return System.identityHashCode(proxy);
                    }
                    if (method.getName().equals("equals")) {
                        return proxy == args[0];
                    }
                    throw new UnsupportedOperationException(method.getName());
                });

        LazyTensor lazy = new LazyTensor(root, backend);

        assertSame(output, lazy.eval());
        assertFalse(root.isEvaluated(), "fusion should materialize the optimized root, not the source root");
        assertEquals(0, outputReleases.get());

        lazy.release();

        assertEquals(1, outputReleases.get(), "fused output must be released exactly once");
        assertEquals(1, queryReleases.get(), "shared query materialization must be released once");
        assertEquals(1, keyReleases.get(), "shared key materialization must be released once");
        assertEquals(1, valueReleases.get(), "shared value materialization must be released once");
        assertEquals(0, leafReleases.get(), "constant leaves remain caller-owned");
        assertFalse(query.isEvaluated());
        assertFalse(key.isEvaluated());
        assertFalse(value.isEvaluated());
    }

    private static GraphNode materializedOperation(Tensor leaf, AtomicInteger releases) {
        GraphNode node = operation(OpType.RELU, List.of(GraphNode.constant(leaf)));
        node.setMaterializedData(tensor(releases));
        return node;
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
                    case "toString" -> "FusionLifecycleTensor";
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }
}
