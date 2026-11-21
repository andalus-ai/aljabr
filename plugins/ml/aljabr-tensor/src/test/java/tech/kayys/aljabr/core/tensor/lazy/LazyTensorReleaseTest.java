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

class LazyTensorReleaseTest {

    @Test
    void releaseClearsRootMaterializationBeforeReleasingIt() {
        AtomicInteger releases = new AtomicInteger();
        Tensor leaf = trackingTensor(new AtomicInteger());
        Tensor output = trackingTensor(releases);
        Shape shape = new Shape(2, 2);
        GraphNode root = new GraphNode(
                OpType.RELU, List.of(GraphNode.constant(leaf)), new Object[0], shape, DType.F32);
        root.setMaterializedData(output);
        LazyTensor lazy = new LazyTensor(root, null);

        lazy.release();

        assertEquals(1, releases.get());
        assertFalse(root.isEvaluated());
        assertTrue(root.getMaterializedData() == null);

        lazy.release();
        assertEquals(1, releases.get());
    }

    @Test
    void releaseDoesNotReleaseBorrowedConstantLeaf() {
        AtomicInteger releases = new AtomicInteger();
        Tensor leaf = trackingTensor(releases);
        LazyTensor lazy = LazyTensor.of(leaf);
        GraphNode leafNode = lazy.getNode();

        lazy.release();

        assertEquals(0, releases.get());
        assertTrue(leafNode.isEvaluated());
    }

    private static Tensor trackingTensor(AtomicInteger releases) {
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
                    case "toString" -> "TrackingTensor";
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }
}
