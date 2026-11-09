package tech.kayys.aljabr.core.tensor;

import java.util.Random;
import tech.kayys.aljabr.core.backend.ComputeBackend;

public final class TensorFactory {
    private static final Random RNG = new Random();
    private TensorFactory() {}

    public static Tensor randn(long... shape) {
        long n = numel(shape);
        float[] data = new float[arrayLength(n)];
        for (int i = 0; i < n; i++) data[i] = (float) RNG.nextGaussian();
        return of(data, shape);
    }
    public static Tensor randn(ComputeBackend backend, long... shape) {
        long n = numel(shape);
        float[] data = new float[arrayLength(n)];
        for (int i = 0; i < n; i++) data[i] = (float) RNG.nextGaussian();
        return of(data, backend, shape);
    }
    public static Tensor zeros(long... shape) { return of(new float[arrayLength(numel(shape))], shape); }
    public static Tensor zeros(ComputeBackend backend, long... shape) {
        return of(new float[arrayLength(numel(shape))], backend, shape);
    }
    public static Tensor ones(long... shape) {
        long n = numel(shape); float[] data = new float[arrayLength(n)];
        java.util.Arrays.fill(data, 1f); return of(data, shape);
    }
    public static Tensor ones(ComputeBackend backend, long... shape) {
        long n = numel(shape); float[] data = new float[arrayLength(n)];
        java.util.Arrays.fill(data, 1f); return of(data, backend, shape);
    }
    public static Tensor full(float value, long... shape) {
        long n = numel(shape); float[] data = new float[arrayLength(n)];
        java.util.Arrays.fill(data, value); return of(data, shape);
    }
    public static Tensor full(float value, ComputeBackend backend, long... shape) {
        long n = numel(shape); float[] data = new float[arrayLength(n)];
        java.util.Arrays.fill(data, value); return of(data, backend, shape);
    }
    public static Tensor of(float[] data, long... shape) { return of(data, findDefaultBackend(), shape); }

    public static Tensor of(float[] data, ComputeBackend backend, long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        java.util.Objects.requireNonNull(backend, "backend must not be null");
        long n = numel(shape);
        if (data.length != n) throw new IllegalArgumentException("data length != shape numel");
        long bytes = Math.multiplyExact(n, (long) Float.BYTES);
        tech.kayys.aljabr.core.memory.TensorStorage storage = allocateStorage(backend, bytes);
        try {
            java.lang.foreign.MemorySegment src = java.lang.foreign.MemorySegment.ofArray(data);
            backend.copyFromHost(storage, src, bytes);
            return new DefaultTensor(new Shape(shape), DType.F32, backend.deviceType(), storage, backend);
        } catch (RuntimeException | Error failure) {
            storage.release(); throw failure;
        }
    }

    public static Tensor ofQuantized(java.lang.foreign.MemorySegment data, DType dtype, long... shape) {
        return ofQuantized(data, findDefaultBackend(), dtype, shape);
    }

    public static Tensor ofQuantized(java.lang.foreign.MemorySegment data, ComputeBackend backend,
            DType dtype, long... shape) {
        java.util.Objects.requireNonNull(data, "data must not be null");
        java.util.Objects.requireNonNull(backend, "backend must not be null");
        java.util.Objects.requireNonNull(dtype, "dtype must not be null");
        if (!dtype.isQuantized()) {
            throw new IllegalArgumentException("ofQuantized requires a quantized dtype; actual dtype is " + dtype);
        }
        long n = numel(shape);
        long bytes = dtype.memoryFootprintBytes(n);
        if (data.byteSize() < bytes) throw new IllegalArgumentException("segment too small for quantized tensor");
        tech.kayys.aljabr.core.memory.TensorStorage storage = allocateStorage(backend, bytes);
        try {
            backend.copyFromHost(storage, data, bytes);
            return new DefaultTensor(new Shape(shape), dtype, backend.deviceType(), storage, backend);
        } catch (RuntimeException | Error failure) {
            storage.release(); throw failure;
        }
    }

    private static tech.kayys.aljabr.core.memory.TensorStorage allocateStorage(
            ComputeBackend backend, long requiredBytes) {
        tech.kayys.aljabr.core.memory.TensorStorage storage = java.util.Objects.requireNonNull(
                backend.allocateStorage(requiredBytes), "backend.allocateStorage must not return null");
        long capacity = storage.sizeBytes();
        if (capacity < requiredBytes) {
            storage.release();
            throw new IllegalStateException("Backend " + backend.getClass().getName()
                    + " allocated " + capacity + " bytes; " + requiredBytes + " required");
        }
        return storage;
    }

    private static long numel(long... shape) { return new Shape(shape).numel(); }
    private static int arrayLength(long numel) {
        if (numel > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Tensor is too large for a Java float[]: " + numel + " elements");
        return Math.toIntExact(numel);
    }

    private static volatile ComputeBackend defaultBackend;
    private static ComputeBackend findDefaultBackend() {
        if (defaultBackend != null) return defaultBackend;
        synchronized (TensorFactory.class) {
            if (defaultBackend != null) return defaultBackend;
            String requested = System.getProperty("aljabr.compute.backend", "auto")
                    .trim().toLowerCase(java.util.Locale.ROOT);
            if (requested.isEmpty()) requested = "auto";
            if ("auto".equals(requested)) {
                try {
                    defaultBackend = instantiateBackend("tech.kayys.aljabr.backend.metal.MetalComputeBackend");
                } catch (RuntimeException | LinkageError metalUnavailable) {
                    try {
                        defaultBackend = instantiateBackend("tech.kayys.aljabr.backend.cpu.CpuBackend");
                    } catch (RuntimeException | LinkageError cpuUnavailable) {
                        cpuUnavailable.addSuppressed(metalUnavailable);
                        throw new IllegalStateException("No Aljabr compute backend is available; install the CPU backend "
                                + "or configure an available backend explicitly", cpuUnavailable);
                    }
                }
                return defaultBackend;
            }
            String backendClass = switch (requested) {
                case "metal" -> "tech.kayys.aljabr.backend.metal.MetalComputeBackend";
                case "cpu" -> "tech.kayys.aljabr.backend.cpu.CpuBackend";
                case "hat" -> "tech.kayys.aljabr.backend.hat.HatComputeBackend";
                default -> throw new IllegalArgumentException("Unknown Aljabr compute backend '" + requested
                        + "'. Supported values: auto, metal, cpu, hat");
            };
            defaultBackend = instantiateBackend(backendClass);
            return defaultBackend;
        }
    }
    private static ComputeBackend instantiateBackend(String className) {
        try {
            Class<?> backendType = Class.forName(className);
            return ComputeBackend.class.cast(backendType.getDeclaredConstructor().newInstance());
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            throw new IllegalStateException("Unable to initialize Aljabr compute backend: " + className, e);
        }
    }
}
