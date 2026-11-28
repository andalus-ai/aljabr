package tech.kayys.aljabr.integration.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.OperationSupportDescriptor;

class TensorCapabilityPlannerTest {

    @Test
    void derivesTensorAndOperationConstraintsBeforeQueryingSession() {
        AtomicReference<Map<String, String>> captured = new AtomicReference<>();
        ProviderSession session = sessionStub(captured);
        Tensor input = tensor(DType.F32, new Shape(2, 3), new TensorLayout(new Shape(2, 3),
                new long[] {3, 1}, 0), DeviceType.CPU);
        Tensor rhs = tensor(DType.F16, new Shape(3, 4), TensorLayout.contiguous(new Shape(3, 4)),
                DeviceType.CPU);

        OperationSupportDescriptor result = TensorCapabilityPlanner.query(
                session, "matmul", List.of(input, rhs), Map.of("transpose_rhs", "false"));

        assertEquals(OperationSupport.NATIVE, result.support());
        Map<String, String> constraints = captured.get();
        assertEquals("F32", constraints.get("dtype"));
        assertEquals("2", constraints.get("rank"));
        assertEquals("2,3", constraints.get("shape"));
        assertEquals("3,1", constraints.get("strides"));
        assertEquals("true", constraints.get("contiguous"));
        assertEquals("0", constraints.get("storage_offset"));
        assertEquals("F16", constraints.get("rhs_dtype"));
        assertEquals("2", constraints.get("rhs_rank"));
        assertEquals("F16", constraints.get("input.1.dtype"));
        assertEquals("false", constraints.get("transpose_rhs"));
    }

    @Test
    void rejectsOperationArgumentsThatWouldOverwriteTensorFacts() {
        ProviderSession session = sessionStub(new AtomicReference<>());
        Tensor input = tensor(DType.F32, new Shape(2), TensorLayout.contiguous(new Shape(2)), DeviceType.CPU);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> TensorCapabilityPlanner.query(session, "normalize", List.of(input),
                        Map.of("dtype", "F16")));

        assertTrue(failure.getMessage().contains("conflicts"));
    }

    @Test
    void rejectsLayoutWhoseShapeDisagreesWithTensorShape() {
        ProviderSession session = sessionStub(new AtomicReference<>());
        Tensor input = tensor(DType.F32, new Shape(2, 3), TensorLayout.contiguous(new Shape(6)), DeviceType.CPU);

        assertThrows(IllegalArgumentException.class,
                () -> TensorCapabilityPlanner.query(session, "reshape", List.of(input), Map.of()));
    }

    private static ProviderSession sessionStub(AtomicReference<Map<String, String>> captured) {
        return (ProviderSession) Proxy.newProxyInstance(
                ProviderSession.class.getClassLoader(),
                new Class<?>[] {ProviderSession.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("operationSupport")) {
                        captured.set((Map<String, String>) args[1]);
                        return new OperationSupportDescriptor((String) args[0], OperationSupport.NATIVE, null);
                    }
                    if (method.getName().equals("device")) {
                        return null;
                    }
                    if (method.getName().equals("computeBackends")) {
                        return Map.of();
                    }
                    return null;
                });
    }

    private static Tensor tensor(DType dtype, Shape shape, TensorLayout layout, DeviceType device) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] {Tensor.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "dtype" -> dtype;
                    case "shape" -> shape;
                    case "layout" -> layout;
                    case "device" -> device;
                    case "toString" -> "Tensor[" + dtype + "," + shape + "]";
                    default -> null;
                });
    }
}
