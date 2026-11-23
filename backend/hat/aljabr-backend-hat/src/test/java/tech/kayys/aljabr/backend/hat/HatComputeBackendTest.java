package tech.kayys.aljabr.backend.hat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.TensorLayout;
import tech.kayys.aljabr.backend.cpu.CpuBackend;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

public class HatComputeBackendTest {

    private HatComputeBackend backend;

    @BeforeEach
    public void setUp() {
        backend = new HatComputeBackend();
    }

    @AfterEach
    public void tearDown() {
        backend.close();
    }

    @Test
    public void closingBackendDoesNotCloseCallerOwnedCpuFallback() {
        CpuBackend cpuFallback = new CpuBackend();
        HatComputeBackend injectedBackend = new HatComputeBackend(cpuFallback);
        injectedBackend.close();

        try (Tensor input = TensorFactory.of(new float[] { 1.0f, 2.0f }, cpuFallback, 2);
                Tensor result = cpuFallback.add(input, input)) {
            assertArrayEquals(new float[] { 2.0f, 4.0f }, result.toFloatArray(), 1e-4f);
        } finally {
            cpuFallback.close();
        }
    }

    @Test
    public void testInitialization() {
        assertNotNull(backend, "HAT backend should be successfully initialized");
    }

    @Test
    public void testFallbackAdd() {
        try (Tensor a = TensorFactory.full(2.0f, 2, 2);
                Tensor b = TensorFactory.full(3.0f, 2, 2);
                Tensor result = backend.add(a, b)) {
            assertNotNull(result);
            assertEquals(4, backend.numel(result));
            assertInstanceOf(CpuBackend.class, result.backend(),
                    "Operations without a HAT kernel must retain CPU fallback ownership");
            assertEquals(tech.kayys.aljabr.core.tensor.DeviceType.CPU, result.device(),
                    "CPU fallback results must report the device that executed the operation");
            for (float value : result.toFloatArray()) {
                assertEquals(5.0f, value, 1e-4);
            }
        }
    }

    @Test
    public void testFallbackMatMul() {
        try (Tensor a = TensorFactory.full(2.0f, 2, 2);
                Tensor b = TensorFactory.full(3.0f, 2, 2);
                Tensor result = backend.matmul(a, b)) {
            assertNotNull(result);
            assertEquals(4, backend.numel(result));
            assertEquals(backend, result.backend(),
                    "HAT results must retain the explicitly selected backend");
            for (float v : result.toFloatArray()) assertEquals(12.0f, v, 1e-4);
        }
    }

    @Test
    public void testCpuFallbackCanConsumeHatOwnedMatmulResult() {
        try (Tensor left = TensorFactory.of(new float[] { 1, 2, 3, 4 }, 2, 2);
                Tensor right = TensorFactory.of(new float[] { 5, 6, 7, 8 }, 2, 2);
                Tensor hatResult = backend.matmul(left, right);
                Tensor fallbackResult = backend.add(hatResult, hatResult)) {
            assertEquals(backend, hatResult.backend());
            assertInstanceOf(CpuBackend.class, fallbackResult.backend());
            assertEquals(tech.kayys.aljabr.core.tensor.DeviceType.CPU, fallbackResult.device());
            assertArrayEquals(new float[] { 38, 44, 86, 100 },
                    fallbackResult.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testHatMatmulCanConsumeCpuFallbackResult() {
        try (Tensor left = TensorFactory.of(new float[] { 1, 2, 3, 4 }, 2, 2);
                Tensor right = TensorFactory.of(new float[] { 5, 6, 7, 8 }, 2, 2);
                Tensor cpuResult = backend.add(left, right);
                Tensor identity = TensorFactory.of(new float[] { 1, 0, 0, 1 }, 2, 2);
                Tensor hatResult = backend.matmul(cpuResult, identity)) {
            assertInstanceOf(CpuBackend.class, cpuResult.backend());
            assertArrayEquals(new float[] { 6, 8, 10, 12 }, cpuResult.toFloatArray(), 1e-4f);
            assertEquals(backend, hatResult.backend());
            assertArrayEquals(new float[] { 6, 8, 10, 12 }, hatResult.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testMatmulSupportsRectangularRowMajorMatrices() {
        try (Tensor a = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
                Tensor b = TensorFactory.of(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2);
                Tensor result = backend.matmul(a, b)) {
            assertEquals(2, result.shape().dim(0));
            assertEquals(2, result.shape().dim(1));
            assertArrayEquals(new float[] { 58, 64, 139, 154 }, result.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testMatmulConsumesStridedCpuFallbackViews() {
        try (Tensor parent = TensorFactory.of(new float[] {
                99.0f, 1.0f, 2.0f, 99.0f,
                99.0f, 3.0f, 4.0f, 99.0f
        }, backend, 2, 4);
                Tensor leftView = backend.slice(parent, new long[] { 0, 1 }, new long[] { 2, 2 });
                Tensor right = TensorFactory.of(new float[] {
                        2.0f, 0.0f,
                        0.0f, 3.0f
                }, backend, 2, 2);
                Tensor result = backend.matmul(leftView, right)) {
            assertInstanceOf(CpuBackend.class, leftView.backend(),
                    "Slice fallback should retain CPU ownership");
            assertEquals(1L, leftView.layout().storageOffset(),
                    "The test input must exercise a non-zero storage offset");
            assertArrayEquals(new long[] { 4, 1 }, leftView.layout().strides(),
                    "The test input must exercise a non-contiguous row stride");
            assertEquals(backend, result.backend(),
                    "Native HAT matmul should own its result after consuming the CPU view");
            assertArrayEquals(new float[] { 2.0f, 6.0f, 6.0f, 12.0f },
                    result.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testMatmulReturnsEmptyOutputsWithoutHATDispatch() {
        try (Tensor left = TensorFactory.of(new float[] {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        }, backend, 2, 3);
                Tensor right = TensorFactory.of(new float[0], backend, 3, 0);
                Tensor result = backend.matmul(left, right)) {
            assertEquals(2, result.shape().dim(0));
            assertEquals(0, result.shape().dim(1));
            assertEquals(0, result.numel());
            assertEquals(backend, result.backend());
            assertArrayEquals(new float[0], result.toFloatArray(), 0.0f);
        }

        try (Tensor left = TensorFactory.of(new float[0], backend, 0, 3);
                Tensor right = TensorFactory.of(new float[] {
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f
                }, backend, 3, 2);
                Tensor result = backend.matmul(left, right)) {
            assertEquals(0, result.shape().dim(0));
            assertEquals(2, result.shape().dim(1));
            assertEquals(0, result.numel());
            assertEquals(backend, result.backend());
            assertArrayEquals(new float[0], result.toFloatArray(), 0.0f);
        }
    }

    @Test
    public void testMatmulConsumesTransposedLogicalLayout() {
        try (Tensor parent = TensorFactory.of(new float[] {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        }, backend, 2, 3)) {
            var storage = parent.storage();
            storage.retain();
            try (Tensor transposed = new DefaultTensor(
                    new Shape(3, 2),
                    DType.F32,
                    parent.device(),
                    storage,
                    backend,
                    new TensorLayout(new Shape(3, 2), new long[] { 1, 3 }, 0));
                    Tensor identity = TensorFactory.of(new float[] {
                            1.0f, 0.0f,
                            0.0f, 1.0f
                    }, backend, 2, 2);
                    Tensor result = backend.matmul(transposed, identity)) {
                assertArrayEquals(new float[] {
                        1.0f, 4.0f,
                        2.0f, 5.0f,
                        3.0f, 6.0f
                }, transposed.toFloatArray(), 1e-4f);
                assertEquals(backend, result.backend());
                assertArrayEquals(new float[] {
                        1.0f, 4.0f,
                        2.0f, 5.0f,
                        3.0f, 6.0f
                }, result.toFloatArray(), 1e-4f);
            }
        }
    }

    @Test
    public void testMatmulReturnsZerosForEmptyReductionDimension() {
        try (Tensor left = TensorFactory.of(new float[0], backend, 2, 0);
                Tensor right = TensorFactory.of(new float[0], backend, 0, 3);
                Tensor result = backend.matmul(left, right)) {
            assertEquals(2, result.shape().dim(0));
            assertEquals(3, result.shape().dim(1));
            assertEquals(backend, result.backend());
            assertArrayEquals(new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                    result.toFloatArray(), 0.0f);
        }
    }

    @Test
    public void testMatmulSupportsRepeatedCallsOnSameBackend() {
        try (Tensor firstLeft = TensorFactory.of(new float[] { 1, 2, 3, 4 }, 2, 2);
                Tensor firstRight = TensorFactory.of(new float[] { 5, 6, 7, 8 }, 2, 2);
                Tensor first = backend.matmul(firstLeft, firstRight);
                Tensor secondLeft = TensorFactory.of(new float[] { 2, 0, 0, 2 }, 2, 2);
                Tensor secondRight = TensorFactory.of(new float[] { 3, 0, 0, 4 }, 2, 2);
                Tensor second = backend.matmul(secondLeft, secondRight)) {
            assertArrayEquals(new float[] { 19, 22, 43, 50 }, first.toFloatArray(), 1e-4f);
            assertArrayEquals(new float[] { 6, 0, 0, 8 }, second.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testMatmulSupportsDifferentShapesAcrossCalls() {
        try (Tensor firstLeft = TensorFactory.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
                Tensor firstRight = TensorFactory.of(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2);
                Tensor first = backend.matmul(firstLeft, firstRight);
                Tensor secondLeft = TensorFactory.of(new float[] { 2, 3, 4, 5 }, 1, 4);
                Tensor secondRight = TensorFactory.of(
                        new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, 4, 3);
                Tensor second = backend.matmul(secondLeft, secondRight)) {
            assertArrayEquals(new float[] { 58, 64, 139, 154 }, first.toFloatArray(), 1e-4f);
            assertArrayEquals(new float[] { 92, 106, 120 }, second.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testMatmulSerializesConcurrentCallsOnSharedBackend() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            List<Future<float[]>> results = new ArrayList<>();
            for (int i = 1; i <= 8; i++) {
                final float scale = i;
                results.add(executor.submit(() -> {
                    try (Tensor left = TensorFactory.of(
                                    new float[] { scale, 0.0f, 0.0f, scale }, 2, 2);
                            Tensor right = TensorFactory.of(
                                    new float[] { 2.0f, 0.0f, 0.0f, 3.0f }, 2, 2);
                            Tensor product = backend.matmul(left, right)) {
                        return product.toFloatArray();
                    }
                }));
            }

            for (int i = 0; i < results.size(); i++) {
                float scale = i + 1.0f;
                assertArrayEquals(
                        new float[] { 2.0f * scale, 0.0f, 0.0f, 3.0f * scale },
                        results.get(i).get(), 1e-4f);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    public void testMatmulRejectsIncompatibleReductionDimensions() {
        try (Tensor a = TensorFactory.ones(2, 3); Tensor b = TensorFactory.ones(4, 2)) {
            assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
        }
    }

    @Test
    public void testMatmulRejectsNonMatrixInputs() {
        try (Tensor a = TensorFactory.ones(3); Tensor b = TensorFactory.ones(3, 2)) {
            assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
        }
    }

    @Test
    public void testDropoutIdentityPreservesSelectedBackendOwnership() {
        try (Tensor input = TensorFactory.of(new float[] { 1.0f, 2.0f }, backend, 2)) {
            assertSame(input, backend.dropout(input, 0.25f, false));
            assertSame(input, backend.dropout(input, 0.0f, true));
        }
    }

    @Test
    public void testRoPECpuFallbackStagesHatOwnedInput() {
        try (Tensor input = TensorFactory.of(
                        new float[] { 1.0f, 0.0f }, backend, 1, 1, 1, 2);
                Tensor result = backend.applyRoPE(input, 1, 10_000.0f, false)) {
            assertEquals(CpuBackend.class, result.backend().getClass(),
                    "CPU RoPE fallback must return a CPU-owned result");
            assertArrayEquals(
                    new float[] { (float) Math.cos(1.0), (float) Math.sin(1.0) },
                    result.toFloatArray(), 1e-5f);
        }
    }



    @Test
    public void testMultiInputCpuFallbacksConsumeHatOwnedHostStorage() {
        try (Tensor layerInput = TensorFactory.of(new float[] { 1.0f, 3.0f }, backend, 1, 2);
                Tensor layerWeight = TensorFactory.of(new float[] { 1.0f, 1.0f }, backend, 2);
                Tensor layerBias = TensorFactory.of(new float[] { 0.0f, 0.0f }, backend, 2);
                Tensor layerResult = backend.layerNorm(
                        layerInput, new long[] { 2 }, layerWeight, layerBias, 1e-5f)) {
            assertInstanceOf(CpuBackend.class, layerResult.backend());
            assertArrayEquals(new float[] { -1.0f, 1.0f }, layerResult.toFloatArray(), 1e-4f);
        }

        try (Tensor rmsInput = TensorFactory.of(new float[] { 3.0f, 4.0f }, backend, 1, 2);
                Tensor rmsWeight = TensorFactory.of(new float[] { 1.0f, 1.0f }, backend, 2);
                Tensor rmsResult = backend.rmsNorm(rmsInput, rmsWeight, 1e-5f)) {
            assertInstanceOf(CpuBackend.class, rmsResult.backend());
            assertArrayEquals(new float[] { 0.6f, 0.8f }, rmsResult.toFloatArray(), 1e-4f);
        }

        try (Tensor batchInput = TensorFactory.of(new float[] { 2.0f, 4.0f }, backend, 1, 2);
                Tensor runningMean = TensorFactory.of(new float[] { 0.0f, 0.0f }, backend, 2);
                Tensor runningVariance = TensorFactory.of(new float[] { 1.0f, 1.0f }, backend, 2);
                Tensor batchResult = backend.batchNorm(
                        batchInput, null, null, runningMean, runningVariance, false, 0.1f, 1e-5f)) {
            assertInstanceOf(CpuBackend.class, batchResult.backend());
            assertArrayEquals(new float[] {
                    (float) (2.0 / Math.sqrt(1.0 + 1e-5)),
                    (float) (4.0 / Math.sqrt(1.0 + 1e-5))
            }, batchResult.toFloatArray(), 1e-4f);
        }

        try (Tensor convInput = TensorFactory.of(
                        new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, backend, 1, 1, 2, 2);
                Tensor convWeight = TensorFactory.of(new float[] { 1.0f }, backend, 1, 1, 1, 1);
                Tensor convBias = TensorFactory.of(new float[] { 0.0f }, backend, 1);
                Tensor convResult = backend.conv2d(convInput, convWeight, convBias, 1, 0, 1, 1)) {
            assertInstanceOf(CpuBackend.class, convResult.backend());
            assertArrayEquals(new float[] { 1.0f, 2.0f, 3.0f, 4.0f },
                    convResult.toFloatArray(), 1e-4f);
        }
    }

    @Test
    public void testEmbeddingAndLossFallbacksConsumeHatOwnedOperands() {
        try (Tensor embeddingWeight = TensorFactory.of(
                        new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f },
                        backend, 3, 2);
                Tensor tokenIds = TensorFactory.of(new int[] { 2, 0 }, backend, 2);
                Tensor embeddingResult = backend.embedding(embeddingWeight, tokenIds, -1)) {
            assertInstanceOf(CpuBackend.class, embeddingResult.backend());
            assertArrayEquals(new float[] { 5.0f, 6.0f, 1.0f, 2.0f },
                    embeddingResult.toFloatArray(), 1e-4f);
        }

        try (Tensor probabilities = TensorFactory.of(new float[] { 0.25f, 0.75f }, backend, 2);
                Tensor labels = TensorFactory.of(new float[] { 0.0f, 1.0f }, backend, 2);
                Tensor loss = backend.binaryCrossEntropy(probabilities, labels)) {
            assertInstanceOf(CpuBackend.class, loss.backend());
            assertEquals((float) (-(Math.log(0.75) + Math.log(0.75)) / 2.0),
                    loss.toFloatArray()[0], 1e-4f);
        }
    }
}
