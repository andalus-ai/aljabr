package tech.kayys.aljabr.core.tensor;

import java.util.Random;
import java.util.Objects;
import java.util.random.RandomGenerator;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.TensorStorage;
import tech.kayys.aljabr.core.memory.TensorAllocator;

/**
 * Factory for creating tensors.
 * This is used by static methods in the {@link Tensor} interface.
 *
 * @author bhangun
 */
public final class TensorFactory {
    private static final Random RNG = new Random();

    private TensorFactory() {
    }

    /** @deprecated Use {@link #randn(ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Tensor randn(long... shape) {
        arrayLength(numel(shape));
        throw implicitBackendSelectionRemoved();
    }

    /**
     * Creates a normally distributed tensor using the factory's default
     * random generator. This overload is convenient for exploratory use;
     * use the overload accepting a {@link RandomGenerator} when reproducible
     * experiments or independent random streams are required.
     */
    public static Tensor randn(ComputeBackend backend, long... shape) {
        return randn(backend, RNG, shape);
    }

    /**
     * Creates a normally distributed tensor using an explicit random generator.
     *
     * <p>The caller owns the generator and therefore controls its seed, state,
     * and stream partitioning. Aljabr does not retain or mutate a global seed.
     * The generator is consumed once per element in row-major order.</p>
     *
     * @param backend compute backend that owns the resulting tensor storage
     * @param randomGenerator caller-owned source of random values
     * @param shape tensor dimensions
     * @return a tensor whose elements follow the generator's Gaussian distribution
     */
    public static Tensor randn(
            ComputeBackend backend,
            RandomGenerator randomGenerator,
            long... shape) {
        Objects.requireNonNull(backend, "backend must not be null");
        Objects.requireNonNull(randomGenerator, "randomGenerator must not be null");
        long n = numel(shape);
        float[] data = new float[arrayLength(n)];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) randomGenerator.nextGaussian();
        }
        return of(data, backend, shape);
    }

    /** @deprecated Use {@link #zeros(ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Tensor zeros(long... shape) {
        arrayLength(numel(shape));
        throw implicitBackendSelectionRemoved();
    }

    /** Creates a zero tensor using an explicitly owned compute backend. */
    public static Tensor zeros(ComputeBackend backend, long... shape) {
        long n = numel(shape);
        return of(new float[arrayLength(n)], backend, shape);
    }

    /** @deprecated Use {@link #ones(ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Tensor ones(long... shape) {
        arrayLength(numel(shape));
        throw implicitBackendSelectionRemoved();
    }

    /** Creates a one tensor using an explicitly owned compute backend. */
    public static Tensor ones(ComputeBackend backend, long... shape) {
        long n = numel(shape);
        float[] data = new float[arrayLength(n)];
        for (int i = 0; i < n; i++) {
            data[i] = 1f;
        }
        return of(data, backend, shape);
    }

    /** @deprecated Use {@link #full(float, ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Tensor full(float value, long... shape) {
        arrayLength(numel(shape));
        throw implicitBackendSelectionRemoved();
    }

    /** Creates a filled tensor using an explicitly owned compute backend. */
    public static Tensor full(float value, ComputeBackend backend, long... shape) {
        long n = numel(shape);
        float[] data = new float[arrayLength(n)];
        for (int i = 0; i < n; i++) {
            data[i] = value;
        }
        return of(data, backend, shape);
    }

    /** @deprecated Use {@link #of(float[], ComputeBackend, long...)} with an owned runtime backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Tensor of(float[] data, long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        long n = numel(shape);
        if (data.length != n) {
            throw new IllegalArgumentException("data length != shape numel");
        }
        throw implicitBackendSelectionRemoved();
    }

    /** Creates a tensor backed by the supplied compute backend. */
    public static Tensor of(float[] data, ComputeBackend backend, long... shape) {
        java.util.Objects.requireNonNull(backend, "backend must not be null");
        return of(data, backend, backend::allocateStorage, shape);
    }

    /**
     * Creates a tensor using an allocation policy independent of the compute
     * backend. The allocator transfers one storage ownership reference to the
     * returned tensor; the backend remains responsible for host-to-storage copy
     * and compute operations.
     */
    public static Tensor of(
            float[] data,
            ComputeBackend backend,
            TensorAllocator allocator,
            long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        java.util.Objects.requireNonNull(backend, "backend must not be null");
        java.util.Objects.requireNonNull(allocator, "allocator must not be null");
        long n = numel(shape);
        if (data.length != n) {
            throw new IllegalArgumentException("data length != shape numel");
        }

        long bytes = Math.multiplyExact(n, (long) Float.BYTES);
        TensorStorage storage = allocateStorage(allocator, bytes);
        try {
            java.lang.foreign.MemorySegment src = java.lang.foreign.MemorySegment.ofArray(data);
            backend.copyFromHost(storage, src, bytes);
            return new DefaultTensor(new Shape(shape), DType.F32, backend.deviceType(), storage, backend);
        } catch (RuntimeException | Error failure) {
            storage.release();
            throw failure;
        }
    }

    /**
     * Deprecated convenience overload retained for source compatibility.
     * It validates its arguments but does not select a backend.
     *
     * @deprecated Pass an explicitly selected, runtime-owned backend.
     */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Tensor ofQuantized(java.lang.foreign.MemorySegment data, DType dtype, long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        java.util.Objects.requireNonNull(dtype, "dtype must not be null");
        if (!dtype.isQuantized()) {
            throw new IllegalArgumentException(
                    "ofQuantized requires a quantized dtype; actual dtype is " + dtype);
        }
        long bytes = dtype.memoryFootprintBytes(numel(shape));
        if (data.byteSize() < bytes) {
            throw new IllegalArgumentException("segment too small for quantized tensor");
        }
        throw implicitBackendSelectionRemoved();
    }

    /**
     * Creates a quantized tensor using the supplied backend's allocation and
     * host-copy contract.
     *
     * <p>The source segment remains owned by the caller. The returned tensor
     * owns its allocated storage.</p>
     */
    public static Tensor ofQuantized(
            java.lang.foreign.MemorySegment data,
            ComputeBackend backend,
            DType dtype,
            long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        java.util.Objects.requireNonNull(backend, "backend must not be null");
        java.util.Objects.requireNonNull(dtype, "dtype must not be null");
        if (!dtype.isQuantized()) {
            throw new IllegalArgumentException(
                    "ofQuantized requires a quantized dtype; actual dtype is " + dtype);
        }

        long n = numel(shape);
        long bytes = dtype.memoryFootprintBytes(n);
        if (data.byteSize() < bytes) {
            throw new IllegalArgumentException("segment too small for quantized tensor");
        }

        return ofQuantized(data, backend, backend::allocateStorage, dtype, shape);
    }

    /**
     * Creates a quantized tensor using an independent storage allocation policy.
     * The source segment remains caller-owned; the returned tensor owns the
     * allocator's storage lease.
     */
    public static Tensor ofQuantized(
            java.lang.foreign.MemorySegment data,
            ComputeBackend backend,
            TensorAllocator allocator,
            DType dtype,
            long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        java.util.Objects.requireNonNull(backend, "backend must not be null");
        java.util.Objects.requireNonNull(allocator, "allocator must not be null");
        java.util.Objects.requireNonNull(dtype, "dtype must not be null");
        if (!dtype.isQuantized()) {
            throw new IllegalArgumentException(
                    "ofQuantized requires a quantized dtype; actual dtype is " + dtype);
        }

        long n = numel(shape);
        long bytes = dtype.memoryFootprintBytes(n);
        if (data.byteSize() < bytes) {
            throw new IllegalArgumentException("segment too small for quantized tensor");
        }

        TensorStorage storage = allocateStorage(allocator, bytes);
        try {
            backend.copyFromHost(storage, data.asSlice(0, bytes), bytes);
            return new DefaultTensor(new Shape(shape), dtype, backend.deviceType(), storage, backend);
        } catch (RuntimeException | Error failure) {
            storage.release();
            throw failure;
        }
    }

    private static TensorStorage allocateStorage(TensorAllocator allocator, long requiredBytes) {
        TensorStorage storage = java.util.Objects.requireNonNull(
                allocator.allocate(requiredBytes),
                "tensor allocator must not return null");
        long availableBytes = storage.sizeBytes();
        if (availableBytes < requiredBytes) {
            storage.release();
            throw new IllegalStateException(
                    "Tensor allocator " + allocator.getClass().getName()
                            + " allocated " + availableBytes + " bytes; "
                            + requiredBytes + " required");
        }
        return storage;
    }

    private static long numel(long... shape) {
        return new Shape(shape).numel();
    }

    private static int arrayLength(long numel) {
        if (numel > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Tensor is too large for a Java float[]: " + numel + " elements");
        }
        return Math.toIntExact(numel);
    }

    /**
     * Fails fast for deprecated convenience methods that previously selected
     * and cached a process-global backend from the tensor core.
     *
     * <p>Backend selection and provider-session ownership belong to the
     * application/runtime. Pass the selected backend explicitly to the
     * overloads that accept {@link ComputeBackend}.</p>
     */
    private static IllegalStateException implicitBackendSelectionRemoved() {
        return new IllegalStateException(
                "Implicit tensor backend selection is no longer supported. "
                        + "Select and own a backend in your application/runtime, then pass it "
                        + "to the TensorFactory overload that accepts ComputeBackend.");
    }
}
