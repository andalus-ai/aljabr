package tech.kayys.aljabr.backend.hat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

        // Matrix multiply of [[2,2],[2,2]] x [[3,3],[3,3]] = [[12,12],[12,12]]
        float[] out = result.toFloatArray();
        for (float v : out) {
            assertEquals(12.0f, v, 1e-4);
        }
    }
}
