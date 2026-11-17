package tech.kayys.aljabr.backend.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.AljabrProvider;
import tech.kayys.aljabr.spi.provider.DeviceKind;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.ProviderContext;
import tech.kayys.aljabr.spi.provider.ProviderException;

class HatComputeProviderTest {
    @Test
    void serviceLoaderDiscoversHatProvider() {
        assertEquals(1, ServiceLoader.load(AljabrProvider.class)
                .stream().map(ServiceLoader.Provider::get)
                .filter(HatComputeProvider.class::isInstance).count());
    }

    @Test
    void providerRequiresInitializationAndOpensHatSession() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        assertEquals("andalus/hat-java", provider.descriptor().id());
        assertEquals(DeviceKind.CPU, provider.devices().get(0).kind());
        assertEquals("hat-java:0", provider.devices().get(0).deviceId());
        assertThrows(ProviderException.class,
                () -> provider.openSession(provider.devices().get(0)));

        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(Set.of(
                    HatComputeProvider.BACKEND_ID,
                    HatComputeProvider.CPU_FALLBACK_BACKEND_ID),
                    session.computeBackends().keySet());
            assertInstanceOf(HatComputeBackend.class,
                    session.computeBackends().get(HatComputeProvider.BACKEND_ID));
        }
        provider.close();
        assertThrows(ProviderException.class, provider::devices);
    }

    @Test
    void operationSupportDistinguishesNativeHatMatmulFromCpuFallback() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));
        try {
            assertEquals(OperationSupport.NATIVE,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F32", "rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F16", "rank", "2")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F32")).support());

            var fallbackUnknown = session.operationSupport("tensor.add", Map.of());
            assertEquals(OperationSupport.UNKNOWN, fallbackUnknown.support());
            var fallback = session.operationSupport("tensor.add", Map.of("dtype", "F32"));
            assertEquals(OperationSupport.DELEGATED, fallback.support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID, fallback.effectiveBackendId());
            assertInstanceOf(CpuBackend.class,
                    session.computeBackends().get(fallback.effectiveBackendId()));
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.reshape", Map.of()).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.reshape", Map.of("dtype", "Q4_0")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.flatten", Map.of()).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.unsqueeze", Map.of()).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.squeeze", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.slice", Map.of()).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.slice", Map.of("dtype", "Q4_0")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.relu", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.cross_entropy", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.cross_entropy",
                            Map.of("dtype", "F32", "target_dtype", "I32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.cross_entropy",
                            Map.of("dtype", "F32", "target_dtype", "F32")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.binary_cross_entropy",
                            Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.binary_cross_entropy",
                            Map.of("dtype", "F32", "target_dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.binary_cross_entropy",
                            Map.of("dtype", "F32", "target_dtype", "I32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.relu", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.relu", Map.of("dtype", "Q4_0")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.layer_norm", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "weight_present", "true", "weight_dtype", "F32",
                            "bias_present", "true", "bias_dtype", "F32")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "weight_present", "true", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "weight_present", "true", "weight_dtype", "F16",
                            "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.rms_norm", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.rms_norm", Map.of(
                            "dtype", "F32", "weight_dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rms_norm", Map.of(
                            "dtype", "F32", "weight_dtype", "Q4_0")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "weight_present", "false", "bias_present", "false"))
                            .support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "4", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "true",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "inference",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "4", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "1", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false",
                            "running_mean_dtype", "F16", "running_var_dtype", "F32",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.conv2d", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.conv2d", Map.of(
                            "dtype", "F32", "rank", "4", "weight_dtype", "F32", "groups", "1",
                            "bias_present", "false", "stride", "1", "dilation", "1", "padding", "0")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.conv2d", Map.of(
                            "dtype", "F32", "rank", "4", "weight_dtype", "F32", "groups", "1",
                            "bias_present", "true", "stride", "1", "dilation", "1", "padding", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.conv2d", Map.of(
                            "dtype", "F32", "rank", "4", "weight_dtype", "F16", "groups", "1",
                            "bias_present", "false", "stride", "1", "dilation", "1", "padding", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.conv2d", Map.of(
                            "dtype", "F32", "rank", "4", "weight_dtype", "F32", "groups", "2",
                            "bias_present", "false", "stride", "1", "dilation", "1", "padding", "0")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.max_pool2d", Map.of(
                            "dtype", "F32", "rank", "4", "kernel_size", "2", "stride", "2", "padding", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.max_pool2d", Map.of(
                            "dtype", "F32", "rank", "3", "kernel_size", "2", "stride", "2", "padding", "0")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.max_pool2d", Map.of("dtype", "F32", "rank", "4")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.adaptive_avg_pool2d", Map.of(
                            "dtype", "F32", "rank", "4", "output_h", "1", "output_w", "1")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.adaptive_avg_pool2d", Map.of(
                            "dtype", "F32", "rank", "4", "output_h", "0", "output_w", "1")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.dropout", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.dropout", Map.of("training", "true", "p", "0.25")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.dropout", Map.of(
                            "training", "false", "p", "0.25", "dtype", "Q4_0")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.dropout", Map.of(
                            "training", "true", "p", "0", "dtype", "Q4_0")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.dropout", Map.of(
                            "training", "true", "p", "0.25", "dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.dropout", Map.of(
                            "training", "true", "p", "0.25", "dtype", "F16")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.dropout", Map.of(
                            "training", "true", "p", "1", "dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.dropout", Map.of(
                            "training", "true", "p", "1.2", "dtype", "F32")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.zeros_like", Map.of()).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.zeros_like", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.zeros_like", Map.of("dtype", "Q4_0")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.embedding", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.cast", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.to", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.cat", Map.of()).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.upsample_nearest2d", Map.of()).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention", Map.of()).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.masked", Map.of()).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rope", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.unknown", Map.of()).support());
        } finally {
            provider.close();
        }
        assertThrows(IllegalStateException.class, session::computeBackends);
    }

    @Test
    void providerCloseClosesOutstandingSessions() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        ProviderSession session = provider.openSession(provider.devices().get(0));

        provider.close();

        assertEquals("hat-java:0", session.device().deviceId());
        assertThrows(IllegalStateException.class, session::computeBackends);
        session.close();
        assertThrows(ProviderException.class, () -> provider.openSession(provider.devices().get(0)));
    }

    @Test
    void sessionExposesOnlyTheProviderOwnedBackendInstance() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            ComputeBackend backend = session.computeBackends().get(HatComputeProvider.BACKEND_ID);
            assertInstanceOf(HatComputeBackend.class, backend);
            assertEquals(2, session.computeBackends().size());
            assertInstanceOf(CpuBackend.class,
                    session.computeBackends().get(HatComputeProvider.CPU_FALLBACK_BACKEND_ID));
        } finally {
            provider.close();
        }
    }
}
