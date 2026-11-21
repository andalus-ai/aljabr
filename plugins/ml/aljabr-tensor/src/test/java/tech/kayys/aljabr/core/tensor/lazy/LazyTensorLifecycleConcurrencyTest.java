package tech.kayys.aljabr.core.tensor.lazy;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LazyTensorLifecycleConcurrencyTest {

    @Test
    void releaseWaitsForInFlightEvaluationAndThenReleasesItsOutput() throws Exception {
        CountDownLatch evaluationEntered = new CountDownLatch(1);
        CountDownLatch allowEvaluationToFinish = new CountDownLatch(1);
        CountDownLatch releaseStarted = new CountDownLatch(1);
        AtomicInteger outputReleases = new AtomicInteger();
        Tensor input = tensor(new AtomicInteger());
        Tensor output = tensor(outputReleases);

        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("relu")) {
                        evaluationEntered.countDown();
                        if (!allowEvaluationToFinish.await(5, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("Timed out waiting to finish test evaluation");
                        }
                        return output;
                    }
                    if (method.getName().equals("toString")) {
                        return "BlockingComputeBackend";
                    }
                    if (method.getName().equals("hashCode")) {
                        return System.identityHashCode(proxy);
                    }
                    if (method.getName().equals("equals")) {
                        return proxy == args[0];
                    }
                    throw new UnsupportedOperationException(method.getName());
                });

        GraphNode root = new GraphNode(
                OpType.RELU,
                List.of(GraphNode.constant(input)),
                new Object[0],
                new Shape(2, 2),
                DType.F32);
        LazyTensor lazy = new LazyTensor(root, backend);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Tensor> evaluation = executor.submit(lazy::eval);
            assertTrue(evaluationEntered.await(5, TimeUnit.SECONDS), "evaluation should enter backend");

            Future<?> release = executor.submit(() -> {
                releaseStarted.countDown();
                lazy.release();
            });
            assertTrue(releaseStarted.await(5, TimeUnit.SECONDS), "release task should start");
            assertThrows(TimeoutException.class, () -> release.get(100, TimeUnit.MILLISECONDS),
                    "release must wait until evaluation has finished");

            allowEvaluationToFinish.countDown();
            assertSame(output, evaluation.get(5, TimeUnit.SECONDS));
            release.get(5, TimeUnit.SECONDS);

            assertFalse(root.isEvaluated(), "release must clear the completed materialization");
            assertEquals(1, outputReleases.get(), "the in-flight output must be released exactly once");
        } finally {
            allowEvaluationToFinish.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void concurrentEvaluationsShareOneMaterialization() throws Exception {
        CountDownLatch evaluationEntered = new CountDownLatch(1);
        CountDownLatch allowEvaluationToFinish = new CountDownLatch(1);
        AtomicInteger backendCalls = new AtomicInteger();
        AtomicInteger outputReleases = new AtomicInteger();
        Tensor output = tensor(outputReleases);
        ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("relu")) {
                        backendCalls.incrementAndGet();
                        evaluationEntered.countDown();
                        if (!allowEvaluationToFinish.await(5, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("Timed out waiting to finish test evaluation");
                        }
                        return output;
                    }
                    if (method.getName().equals("toString")) {
                        return "BlockingComputeBackend";
                    }
                    if (method.getName().equals("hashCode")) {
                        return System.identityHashCode(proxy);
                    }
                    if (method.getName().equals("equals")) {
                        return proxy == args[0];
                    }
                    throw new UnsupportedOperationException(method.getName());
                });

        GraphNode root = new GraphNode(
                OpType.RELU,
                List.of(GraphNode.constant(tensor(new AtomicInteger()))),
                new Object[0],
                new Shape(2, 2),
                DType.F32);
        LazyTensor lazy = new LazyTensor(root, backend);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Tensor> first = executor.submit(lazy::eval);
            assertTrue(evaluationEntered.await(5, TimeUnit.SECONDS), "first evaluation should enter backend");
            Future<Tensor> second = executor.submit(lazy::eval);

            allowEvaluationToFinish.countDown();
            assertSame(output, first.get(5, TimeUnit.SECONDS));
            assertSame(output, second.get(5, TimeUnit.SECONDS));
            assertEquals(1, backendCalls.get(), "only one backend evaluation should run");

            lazy.release();
            assertEquals(1, outputReleases.get());
        } finally {
            allowEvaluationToFinish.countDown();
            executor.shutdownNow();
        }
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
                    case "toString" -> "LifecycleTrackingTensor";
                    default -> throw new UnsupportedOperationException(method.getName());
                });
    }
}
