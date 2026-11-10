package tech.kayys.aljabr.spi.provider;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class ProviderValueTypesTest {
    @Test void deviceDescriptorNormalizesOptionalMetadataAndPreservesUnknownMemory() {
        var device = new DeviceDescriptor(
                "andalus/cpu-reference", "socket-0", " Reference CPU ", DeviceKind.CPU,
                null, null, " Example Vendor ", " x86_64 ", null, " 1.0 ");

        assertEquals("Reference CPU", device.name());
        assertNull(device.totalMemoryBytes());
        assertNull(device.freeMemoryBytes());
        assertEquals("Example Vendor", device.vendor());
        assertEquals("x86_64", device.architecture());
        assertNull(device.driverVersion());
        assertEquals("1.0", device.runtimeVersion());
    }

    @Test void deviceDescriptorRejectsInvalidMemoryFacts() {
        assertThrows(IllegalArgumentException.class, () -> new DeviceDescriptor(
                "andalus/cpu-reference", "cpu-0", "CPU", DeviceKind.CPU, -1L, null,
                null, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new DeviceDescriptor(
                "andalus/cpu-reference", "gpu-0", "GPU", DeviceKind.GPU, 1024L, 2048L,
                null, null, null, null));
    }

    @Test void deviceDescriptorRequiresProviderLocalIdentityAndKind() {
        assertThrows(IllegalArgumentException.class, () -> new DeviceDescriptor(
                " ", "cpu-0", "CPU", DeviceKind.CPU, null, null, null, null, null, null));
        assertThrows(NullPointerException.class, () -> new DeviceDescriptor(
                "andalus/cpu-reference", "cpu-0", "CPU", null, null, null, null, null, null, null));
    }

    @Test void delegatedOperationMustIdentifyEffectiveBackend() {
        assertThrows(IllegalArgumentException.class, () -> new OperationSupportDescriptor(
                "tensor.matmul", OperationSupport.DELEGATED, null));
        var delegated = new OperationSupportDescriptor(
                " tensor.matmul ", OperationSupport.DELEGATED, " cpu/reference ");
        assertEquals("tensor.matmul", delegated.operationId());
        assertEquals("cpu/reference", delegated.effectiveBackendId());
    }

    @Test void nativeAndUnknownSupportDoNotRequireEffectiveBackend() {
        assertNull(new OperationSupportDescriptor(
                "tensor.add", OperationSupport.NATIVE, null).effectiveBackendId());
        assertNull(new OperationSupportDescriptor(
                "tensor.custom", OperationSupport.UNKNOWN, null).effectiveBackendId());
    }

    @Test void readinessFactsRemainIndependentValues() {
        assertNotEquals(ProviderReadiness.INITIALIZED, ProviderReadiness.VALIDATED);
        assertEquals(8, ProviderReadiness.values().length);
    }
}
