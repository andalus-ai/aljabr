package tech.kayys.aljabr.backend.hat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class HatComputeBackendTest {

    private HatComputeBackend backend;

    @BeforeEach
    public void setUp() {
        backend = new HatComputeBackend();
    }

    @AfterEach
    public void tearDown() {
        // ComputeBackend interface does not currently define a close() method
    }

    @Test
    public void testInitialization() {
        assertNotNull(backend, "HAT backend should be successfully initialized");
    }

    @Test
    public void testFallbackAdd() {
        // Create simple 2x2 tensors using the factory (which may or may not map to HAT
        // yet depending on the registry,
        // but we can test the backend directly for its fallback behavior).
        Tensor a = TensorFactory.full(2.0f, 2, 2);
        Tensor b = TensorFactory.full(3.0f, 2, 2);

        // Use our HAT backend directly to invoke the add
        Tensor result = backend.add(a, b);

        assertNotNull(result);
        assertEquals(4, backend.numel(result));

        // Verify values via array copy (CPU mode mapping)
        float[] out = result.toFloatArray();
        for (float v : out) {
            assertEquals(5.0f, v, 1e-4);
        }
    }

    @Test
    public void testFallbackMatMul() {
        Tensor a = TensorFactory.full(2.0f, 2, 2); // 2x2 matrix of 2s
        Tensor b = TensorFactory.full(3.0f, 2, 2); // 2x2 matrix of 3s

        Tensor result = backend.matmul(a, b);

        assertNotNull(result);
        assertEquals(4, backend.numel(result));
        assertEquals(backend, result.backend(), "HAT results must retain the explicitly selected backend");

        // Matrix multiply of [[2,2],[2,2]] x [[3,3],[3,3]] = [[12,12],[12,12]]
        float[] out = result.toFloatArray();
        for (float v : out) {
            assertEquals(12.0f, v, 1e-4);
        }
    }
    @Test
    public void testMatmulSupportsRectangularRowMajorMatrices() {
        Tensor a = TensorFactory.of(new float[] {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        }, 2, 3);
        Tensor b = TensorFactory.of(new float[] {
                7.0f, 8.0f,
                9.0f, 10.0f,
                11.0f, 12.0f
        }, 3, 2);

        Tensor result = backend.matmul(a, b);

        assertEquals(2, result.shape().dim(0));
        assertEquals(2, result.shape().dim(1));
        assertArrayEquals(new float[] { 58.0f, 64.0f, 139.0f, 154.0f },
                result.toFloatArray(), 1e-4f);
    }

    @Test
    public void testMatmulSupportsRepeatedCallsOnSameBackend() {
        Tensor first = backend.matmul(
                TensorFactory.of(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, 2, 2),
                TensorFactory.of(new float[] { 5.0f, 6.0f, 7.0f, 8.0f }, 2, 2));
        Tensor second = backend.matmul(
                TensorFactory.of(new float[] { 2.0f, 0.0f, 0.0f, 2.0f }, 2, 2),
                TensorFactory.of(new float[] { 3.0f, 0.0f, 0.0f, 4.0f }, 2, 2));

        assertArrayEquals(new float[] { 19.0f, 22.0f, 43.0f, 50.0f },
                first.toFloatArray(), 1e-4f);
        assertArrayEquals(new float[] { 6.0f, 0.0f, 0.0f, 8.0f },
                second.toFloatArray(), 1e-4f);
    }

    @Test
    public void testMatmulSupportsDifferentShapesAcrossCalls() {
        Tensor first = backend.matmul(
                TensorFactory.of(new float[] {
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                }, 2, 3),
                TensorFactory.of(new float[] {
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                }, 3, 2));
        Tensor second = backend.matmul(
                TensorFactory.of(new float[] { 2.0f, 3.0f, 4.0f, 5.0f }, 1, 4),
                TensorFactory.of(new float[] {
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f,
                        10.0f, 11.0f, 12.0f
                }, 4, 3));

        assertArrayEquals(new float[] { 58.0f, 64.0f, 139.0f, 154.0f },
                first.toFloatArray(), 1e-4f);
        assertArrayEquals(new float[] { 92.0f, 106.0f, 120.0f },
                second.toFloatArray(), 1e-4f);
    }

    @Test
    public void testMatmulSerializesConcurrentCallsOnSharedBackend() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            List<Future<float[]>> results = new ArrayList<>();
            for (int i = 1; i <= 8; i++) {
                final float scale = i;
                results.add(executor.submit(() -> {
                    Tensor left = TensorFactory.of(
                            new float[] { scale, 0.0f, 0.0f, scale }, 2, 2);
                    Tensor right = TensorFactory.of(
                            new float[] { 2.0f, 0.0f, 0.0f, 3.0f }, 2, 2);
                    return backend.matmul(left, right).toFloatArray();
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
        Tensor a = TensorFactory.ones(2, 3);
        Tensor b = TensorFactory.ones(4, 2);

        assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
    }

    @Test
    public void testMatmulRejectsNonMatrixInputs() {
        Tensor a = TensorFactory.ones(3);
        Tensor b = TensorFactory.ones(3, 2);

        assertThrows(IllegalArgumentException.class, () -> backend.matmul(a, b));
    }

}
