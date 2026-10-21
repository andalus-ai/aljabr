package tech.kayys.aljabr.core.tensor;

import java.util.Random;
import tech.kayys.aljabr.core.backend.ComputeBackend;

/**
 * Factory for creating tensors.
 * This is used by static methods in the {@link Tensor} interface.
 * @author bhangun
 */
public final class TensorFactory {
    private static final Random RNG = new Random();

    public static Tensor randn(long... shape) {
        long n = numel(shape);
        float[] data = new float[(int) n];
        for (int i = 0; i < n; i++)
            data[i] = (float) (RNG.nextGaussian());
        return of(data, shape);
    }

    public static Tensor zeros(long... shape) {
        long n = numel(shape);
        float[] data = new float[(int) n]; // zero-initialized
        return of(data, shape);
    }

    public static Tensor ones(long... shape) {
        long n = numel(shape);
        float[] data = new float[(int) n];
        for (int i = 0; i < n; i++)
            data[i] = 1f;
        return of(data, shape);
    }

    public static Tensor full(float value, long... shape) {
        long n = numel(shape);
        float[] data = new float[(int) n];
        for (int i = 0; i < n; i++)
            data[i] = value;
        return of(data, shape);
    }

    public static Tensor of(float[] data, long... shape) {
        long n = numel(shape);
        if (data.length != n)
            throw new IllegalArgumentException("data length != shape numel");

        // Try to instantiate a CPU backend if available via reflection, otherwise use a
        // noop backend
        ComputeBackend backend = findDefaultBackend();

        // Allocate an off-heap buffer and copy data
        long bytes = n * 4L;
        tech.kayys.aljabr.core.memory.CpuBuffer buf = new tech.kayys.aljabr.core.memory.CpuBuffer(bytes);
        java.lang.foreign.MemorySegment src = java.lang.foreign.MemorySegment.ofArray(data);
        buf.segment().copyFrom(src);

        return new DefaultTensor(new Shape(shape), DType.F32, DeviceType.CPU, buf, backend);
    }

    public static Tensor ofQuantized(java.lang.foreign.MemorySegment data, DType dtype, long... shape) {
        long n = numel(shape);
        long bytes = (n / dtype.blockSize()) * dtype.blockByteSize();
        if (data.byteSize() < bytes)
            throw new IllegalArgumentException("segment too small for quantized tensor");

        ComputeBackend backend = findDefaultBackend();
        tech.kayys.aljabr.core.memory.CpuBuffer buf = new tech.kayys.aljabr.core.memory.CpuBuffer(data, null);
        return new DefaultTensor(new Shape(shape), dtype, DeviceType.CPU, buf, backend);
    }

    private static long numel(long... shape) {
        long n = 1;
        for (long s : shape)
            n *= s;
        return n;
    }

    private static volatile ComputeBackend defaultBackend;

    private static ComputeBackend findDefaultBackend() {
        if (defaultBackend != null)
            return defaultBackend;
        synchronized (TensorFactory.class) {
            if (defaultBackend != null)
                return defaultBackend;
            try {
                // Try HAT Tensor Backend first (Hardware accelerated via OpenJDK Babylon)
                Class<?> c = Class.forName("tech.kayys.aljabr.backend.hat.HatComputeBackend");
                defaultBackend = (ComputeBackend) c.getDeclaredConstructor().newInstance();
            } catch (Throwable tHat) {
                try {
                    // Fallback to Metal native bindings
                    Class<?> c = Class.forName("tech.kayys.aljabr.backend.metal.MetalComputeBackend");
                    defaultBackend = (ComputeBackend) c.getDeclaredConstructor().newInstance();
                } catch (Throwable tMetal) {
                    try {
                        // Fallback to CPU backend
                        Class<?> c = Class.forName("tech.kayys.aljabr.backend.cpu.CpuBackend");
                        defaultBackend = (ComputeBackend) c.getDeclaredConstructor().newInstance();
                    } catch (Throwable tCpu) {
                        // Fallback backend that throws on use
                        defaultBackend = new ComputeBackend() {
                            private UnsupportedOperationException u() {
                                return new UnsupportedOperationException("No backend available");
                            }

                            @Override
                            public Tensor add(Tensor a, Tensor b) {
                                throw u();
                            }

                            @Override
                            public Tensor sub(Tensor a, Tensor b) {
                                throw u();
                            }

                            @Override
                            public Tensor mul(Tensor a, Tensor b) {
                                throw u();
                            }

                            @Override
                            public Tensor div(Tensor a, Tensor b) {
                                throw u();
                            }

                            @Override
                            public Tensor mul(Tensor a, float scalar) {
                                throw u();
                            }

                            @Override
                            public Tensor div(Tensor a, float scalar) {
                                throw u();
                            }

                            @Override
                            public Tensor addScalar(Tensor a, float scalar) {
                                throw u();
                            }

                            @Override
                            public Tensor matmul(Tensor a, Tensor b) {
                                throw u();
                            }

                            @Override
                            public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
                                throw u();
                            }

                            @Override
                            public Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
                                throw u();
                            }

                            @Override
                            public Tensor reshape(Tensor a, long... newShape) {
                                throw u();
                            }

                            @Override
                            public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
                                throw u();
                            }

                            @Override
                            public java.util.List<Tensor> split(Tensor a, int axis, int parts) {
                                throw u();
                            }

                            @Override
                            public Tensor flatten(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor unsqueeze(Tensor a, int dim) {
                                throw u();
                            }

                            @Override
                            public Tensor squeeze(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor transpose(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor transpose(Tensor a, int dim0, int dim1) {
                                throw u();
                            }

                            @Override
                            public Tensor softmax(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor softmax(Tensor a, int dim) {
                                throw u();
                            }

                            @Override
                            public Tensor logSoftmax(Tensor a, int dim) {
                                throw u();
                            }

                            @Override
                            public Tensor mean(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor mean(Tensor a, int dim, boolean keepDim) {
                                throw u();
                            }

                            @Override
                            public Tensor sum(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor sum(Tensor a, int dim, boolean keepDim) {
                                throw u();
                            }

                            @Override
                            public Tensor max(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor abs(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor relu(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor gelu(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor silu(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor sigmoid(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor tanh(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor sqrt(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor exp(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor log(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor pow(Tensor a, float exponent) {
                                throw u();
                            }

                            @Override
                            public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias,
                                    float eps) {
                                throw u();
                            }

                            @Override
                            public Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
                                throw u();
                            }

                            @Override
                            public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias, Tensor runningMean,
                                    Tensor runningVar, boolean training, float momentum, float eps) {
                                throw u();
                            }

                            @Override
                            public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding,
                                    int dilation, int groups) {
                                throw u();
                            }

                            @Override
                            public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
                                throw u();
                            }

                            @Override
                            public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
                                throw u();
                            }

                            @Override
                            public Tensor dropout(Tensor input, float p, boolean training) {
                                throw u();
                            }

                            @Override
                            public Tensor crossEntropy(Tensor pred, Tensor target) {
                                throw u();
                            }

                            @Override
                            public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
                                throw u();
                            }

                            @Override
                            public Tensor attention(Tensor Q, Tensor K, Tensor V) {
                                throw u();
                            }

                            @Override
                            public Tensor zerosLike(Tensor a) {
                                throw u();
                            }

                            @Override
                            public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
                                throw u();
                            }

                            @Override
                            public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
                                throw u();
                            }

                            @Override
                            public long numel(Tensor a) {
                                throw u();
                            }
                        };
                    }
                }
            }
            return defaultBackend;
        }
    }
}
