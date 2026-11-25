package tech.kayys.aljabr.testing;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Reusable conformance checks for tensor slice views.
 *
 * <p>Provider test suites supply a fresh compute backend. The contract checks
 * the public layout and ownership semantics rather than backend internals.</p>
 */
public abstract class TensorViewContract {

    /** Returns a fresh backend instance for the provider under test. */
    protected abstract ComputeBackend createBackend();

    @Test
    public final void slicePreservesPhysicalLayoutAndOwnsStorageIndependently() {
        ComputeBackend backend = createBackend();
        Tensor source = TensorFactory.of(
                new float[] {0, 1, 2, 3, 4, 5},
                backend,
                2,
                3);
        Tensor slice = null;
        try {
            TensorStorage sourceStorage = source.storage();
            slice = source.slice(new long[] {1, 1}, new long[] {1, 2});

            assertEquals(new Shape(1, 2), slice.shape());
            assertArrayEquals(new long[] {3, 1}, slice.layout().strides());
            assertEquals(4L, slice.layout().storageOffset());
            assertSame(sourceStorage, slice.storage(),
                    "a slice view must refer to the source storage");
            assertTrue(sourceStorage.sizeBytes() >= 24L);

            // The slice owns a retained reference and must remain valid after
            // the source relinquishes its own reference.
            source.release();
            assertEquals(new Shape(1, 2), slice.shape());
            assertSame(sourceStorage, slice.storage());
            assertTrue(slice.storage().sizeBytes() >= 24L);
        } finally {
            if (slice != null) {
                slice.release();
            }
            source.release();
            backend.close();
        }
    }
    @Test
    public final void reductionsReadLogicalElementsFromStridedSlices() {
        ComputeBackend backend = createBackend();
        Tensor source = TensorFactory.of(
                new float[] {0, 1, 2, 3, 4, 5},
                backend,
                2,
                3);
        Tensor slice = null;
        Tensor sum = null;
        try {
            slice = source.slice(new long[] {1, 1}, new long[] {1, 2});
            sum = slice.sum();

            // The view addresses source elements 4 and 5, not the first two
            // elements of its underlying storage.
            assertEquals(9.0f, sum.item(), 1.0e-6f);
        } finally {
            if (sum != null) {
                sum.release();
            }
            if (slice != null) {
                slice.release();
            }
            source.release();
            backend.close();
        }
    }

    @Test
    public final void contiguousMaterializationCopiesLogicalValuesIntoIndependentStorage() {
        ComputeBackend backend = createBackend();
        Tensor source = TensorFactory.of(
                new float[] {0, 1, 2, 3, 4, 5}, backend, 2, 3);
        Tensor view = null;
        Tensor contiguous = null;
        try {
            view = source.slice(new long[] {1, 1}, new long[] {1, 2});
            contiguous = view.contiguous();

            assertEquals(new Shape(1, 2), contiguous.shape());
            assertArrayEquals(new long[] {2, 1}, contiguous.layout().strides());
            assertEquals(0L, contiguous.layout().storageOffset());
            assertTrue(contiguous.layout().isContiguous());
            org.junit.jupiter.api.Assertions.assertNotSame(
                    view.storage(), contiguous.storage(),
                    "materialization must allocate independent storage");
            assertArrayEquals(new float[] {4f, 5f}, contiguous.toFloatArray());
            assertArrayEquals(new float[] {4f, 5f}, view.toFloatArray());
        } finally {
            if (contiguous != null) contiguous.release();
            if (view != null) view.release();
            source.release();
            backend.close();
        }
    }


    @Test
    public final void siblingViewsRetainSharedStorageIndependently() {
        ComputeBackend backend = createBackend();
        Tensor source = TensorFactory.of(
                new float[] {0, 1, 2, 3, 4, 5}, backend, 2, 3);
        Tensor firstView = null;
        Tensor secondView = null;
        try {
            TensorStorage sharedStorage = source.storage();
            firstView = source.slice(new long[] {0, 0}, new long[] {1, 3});
            secondView = source.slice(new long[] {1, 0}, new long[] {1, 3});

            assertSame(sharedStorage, firstView.storage());
            assertSame(sharedStorage, secondView.storage());

            // Each view owns a reference to the shared allocation. Releasing
            // the source and one view must not invalidate the sibling view.
            source.release();
            firstView.release();
            firstView = null;

            assertSame(sharedStorage, secondView.storage());
            assertEquals(new Shape(1, 3), secondView.shape());
            assertArrayEquals(new float[] {3f, 4f, 5f}, secondView.toFloatArray());
        } finally {
            if (secondView != null) secondView.release();
            if (firstView != null) firstView.release();
            source.release();
            backend.close();
        }
    }

}
