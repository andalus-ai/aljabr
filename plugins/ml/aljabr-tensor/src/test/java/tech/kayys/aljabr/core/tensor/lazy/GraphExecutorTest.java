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
import java.util.concurrent.atomic.AtomicReference;

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

    @Test
    void releaseAllAttemptsEveryTensorAndSuppressesLaterFailures() {
        AtomicInteger firstAttempts = new AtomicInteger();
        AtomicInteger secondAttempts = new AtomicInteger();
        AtomicInteger thirdAttempts = new AtomicInteger();
        IllegalStateException firstFailure = new IllegalStateException("first cleanup failure");
        IllegalArgumentException secondFailure = new IllegalArgumentException("second cleanup failure");

        Tensor first = releaseFailingTensor(firstAttempts, firstFailure);
        Tensor second = releaseFailingTensor(secondAttempts, secondFailure);
        Tensor third = releaseFailingTensor(thirdAttempts, null);

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> GraphExecutor.releaseAll(first, second, third));

        assertSame(firstFailure, thrown);
        assertEquals(1, thrown.getSuppressed().length);
        assertSame(secondFailure, thrown.getSuppressed()[0]);
        assertEquals(1, firstAttempts.get());
        assertEquals(1, secondAttempts.get());
        assertEquals(1, thirdAttempts.get());
    }

    @Test
    void leafGradientIsReleasedWhenReplacingPreviousGradientFails() {
        Shape shape = new Shape(2, 2);
        AtomicInteger propagatedGradientReleases = new AtomicInteger();
        AtomicInteger accumulatedGradientReleases = new AtomicInteger();
        Tensor propagatedGradient = tensor(propagatedGradientReleases);
        Tensor accumulatedGradient = tensorWithAdd(shape, accumulatedGradientReleases);
        IllegalStateException replacementFailure = new IllegalStateException("previous gradient release failed");
        AtomicReference<Tensor> leafOwnedGradient = new AtomicReference<>();

        AtomicInteger leafGradientReleases = new AtomicInteger();
        Tensor leaf = (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "shape" -> shape;
                    case "dtype" -> DType.F32;
                    case "device" -> DeviceType.CPU;
                    case "requiresGrad" -> true;
                    case "grad" -> accumulatedGradient;
                    case "setGrad" -> {
                        leafOwnedGradient.set((Tensor) args[0]);
                        throw replacementFailure;
                    }
                    case "toString" -> "DifferentiableLeaf";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> throw new UnsupportedOperationException(method.getName());
                });

        AtomicInteger seedReleases = new AtomicInteger();
        Tensor seed = tensorWithScalarAdd(shape, seedReleases);
        AtomicInteger outputReleases = new AtomicInteger();
        Tensor output = tensor(outputReleases);
        GraphNode leafNode = GraphNode.constant(leaf);
        GraphNode root = new GraphNode(
                OpType.RELU,
                List.of(leafNode),
                new Object[0],
                shape,
                DType.F32,
                (forward, outputGradient, inputs, arguments) ->
                        new Tensor[] { propagatedGradient });
        root.setMaterializedData(output);

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> new GraphExecutor(null).backward(root, seed));

        assertSame(replacementFailure, thrown);
        assertSame(leafOwnedGradient.get(), leafOwnedGradient.get(),
                "the replacement gradient is retained by the leaf after ownership transfer");
        assertEquals(1, propagatedGradientReleases.get(),
                "the executor-owned gradient must be released after setGrad fails");
        assertEquals(1, seedReleases.get(),
                "the output seed copy must be released after the backward failure");
        assertEquals(0, accumulatedGradientReleases.get(),
                "the replacement gradient remains leaf-owned after setGrad transfers ownership");
        assertEquals(0, outputReleases.get(), "materialized forward output remains caller/graph-owned");
    }

    private static Tensor tensorWithScalarAdd(Shape shape, AtomicInteger releases) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "shape" -> shape;
                    case "add" -> tensor(releases);
                    case "release" -> {
                        releases.incrementAndGet();
                        yield null;
                    }
                    case "toString" -> "SeedTensor";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }

    private static Tensor tensorWithAdd(Shape shape, AtomicInteger resultReleases) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "shape" -> shape;
                    case "add" -> tensor(resultReleases);
                    case "release" -> null;
                    case "toString" -> "PreviousGradient";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }

    private static Tensor releaseFailingTensor(
            AtomicInteger attempts, RuntimeException failure) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("release")) {
                        attempts.incrementAndGet();
                        if (failure != null) {
                            throw failure;
                        }
                        return null;
                    }
                    if (method.getName().equals("toString")) {
                        return "ReleaseFailingTensor";
                    }
                    if (method.getName().equals("hashCode")) {
                        return System.identityHashCode(proxy);
                    }
                    if (method.getName().equals("equals")) {
                        return proxy == args[0];
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
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
