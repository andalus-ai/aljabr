package tech.kayys.aljabr.testing.tensor;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import org.junit.jupiter.api.Assertions;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Reusable lifecycle assertions for tensor-provider and backend test suites.
 *
 * <p>These checks exercise only guarantees expressed by the public
 * {@link Tensor} contract. Providers should add implementation-specific tests
 * for view aliasing, asynchronous completion, device fences, and storage
 * reference counts where those capabilities are exposed.</p>
 */
public final class TensorLifecycleAssertions {

    private TensorLifecycleAssertions() {
    }

    /**
     * Verifies that releasing a tensor repeatedly is harmless and that
     * {@link Tensor#close()} remains a safe alias for release.
     */
    public static void assertReleaseIsIdempotent(Supplier<? extends Tensor> tensorFactory) {
        Tensor tensor = requireTensor(tensorFactory, "tensorFactory");
        try {
            assertDoesNotThrow(tensor::release, "The first release must succeed");
            assertDoesNotThrow(tensor::release, "Repeated release must be harmless");
            assertDoesNotThrow(tensor::close, "close() after release must be harmless");
        } finally {
            tensor.close();
        }
    }

    /**
     * Verifies that ownership transferred through {@link Tensor#setGrad(Tensor)}
     * is discharged exactly once when the owner is closed.
     *
     * <p>The supplied gradient is wrapped only to observe calls to release;
     * all other tensor operations delegate to the original instance.</p>
     */
    public static void assertOwnedGradientReleasedExactlyOnce(
            Supplier<? extends Tensor> ownerFactory,
            Supplier<? extends Tensor> gradientFactory) {
        Tensor owner = requireTensor(ownerFactory, "ownerFactory");
        Tensor gradient = requireTensor(gradientFactory, "gradientFactory");
        AtomicInteger releaseCalls = new AtomicInteger();
        Tensor trackedGradient = trackRelease(gradient, releaseCalls);
        boolean transferred = false;
        try {
            Assertions.assertNotSame(owner, gradient,
                    "An owner cannot also be the gradient instance");
            owner.setGrad(trackedGradient);
            transferred = true;
            assertSame(trackedGradient, owner.grad(),
                    "The owner must expose the transferred gradient as a borrowed reference");

            owner.close();
            assertEquals(1, releaseCalls.get(),
                    "Closing the owner must release its transferred gradient exactly once");
            assertDoesNotThrow(owner::close, "Closing the owner repeatedly must be harmless");
            assertEquals(1, releaseCalls.get(),
                    "Repeated owner close must not release the gradient again");
        } finally {
            try {
                owner.close();
            } finally {
                if (!transferred) {
                    trackedGradient.release();
                }
            }
        }
    }

    private static Tensor requireTensor(Supplier<? extends Tensor> factory, String name) {
        if (factory == null) {
            throw new AssertionError(name + " must not be null");
        }
        Tensor tensor = factory.get();
        if (tensor == null) {
            throw new AssertionError(name + " returned null");
        }
        return tensor;
    }

    private static Tensor trackRelease(Tensor delegate, AtomicInteger releaseCalls) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("release") && method.getParameterCount() == 0) {
                        releaseCalls.incrementAndGet();
                    }
                    try {
                        return method.invoke(delegate, args);
                    } catch (InvocationTargetException failure) {
                        throw failure.getCause();
                    }
                });
    }
}
