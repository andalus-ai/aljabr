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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GraphExecutorTest {

    @Test
    void failedEvaluationReleasesOnlyOutputsCreatedByThatAttempt() {
        AtomicInteger leftReleases = new AtomicInteger();
        AtomicInteger rightReleases = new AtomicInteger();
        Tensor left = tensor(leftReleases);
        Tensor right = tensor(rightReleases);
        AtomicInteger intermediateReleases = new AtomicInteger();
        Tensor intermediate = tensor(intermediateReleases);
        AtomicInteger cachedReleases = new AtomicInteger();
        Tensor cached = tensor(cachedReleases);

        GraphNode add = new GraphNode(
                OpType.ADD, List.of(GraphNode.constant(left), GraphNode.constant(right)),
                new Object[0], new Shape(2, 2), DType.F32);
        GraphNode previouslyMaterialized = new GraphNode(
                OpType.RELU, List.of(GraphNode.constant(left)),
                new Object[0], new Shape(2, 2), DType.F32);
        previouslyMaterialized.setMaterializedData(cached);

        GraphNode root = new GraphNode(
                OpType.MATMUL, List.of(add, previouslyMaterialized),
                new Object[0], new Shape(2, 2), DType.F32);

        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "add" -> intermediate;
                    case "matmul" -> throw new IllegalStateException("backend failure");
                    default -> throw new UnsupportedOperationException(method.getName());
                });

        GraphExecutor executor = new GraphExecutor(backend);
        assertThrows(IllegalStateException.class, () -> executor.evaluate(root));

        assertFalse(add.isEvaluated(), "failed attempt must clear its intermediate node");
        assertFalse(root.isEvaluated(), "failed root must remain unevaluated");
        assertEquals(1, intermediateReleases.get(), "new intermediate must be released once");
        assertTrue(previouslyMaterialized.isEvaluated(), "pre-existing cached output must be preserved");
        assertSame(cached, previouslyMaterialized.getMaterializedData());
        assertEquals(0, cachedReleases.get(), "pre-existing cached output must not be released");
        assertEquals(0, leftReleases.get(), "caller-owned left leaf must not be released");
        assertEquals(0, rightReleases.get(), "caller-owned right leaf must not be released");
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
                    case "toString" -> "TrackingTensor";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }

}
