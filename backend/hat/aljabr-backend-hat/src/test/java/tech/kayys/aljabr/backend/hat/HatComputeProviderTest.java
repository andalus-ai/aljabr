package tech.kayys.aljabr.backend.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
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
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.matmul", Map.of("dtype", "F32", "rank", "2")).support());
            assertEquals(OperationSupport.NATIVE,
                    session.operationSupport("tensor.matmul", Map.of(
                            "dtype", "F32", "rank", "2", "rhs_dtype", "F32", "rhs_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.matmul", Map.of(
                            "dtype", "F16", "rank", "2", "rhs_dtype", "F32", "rhs_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.matmul", Map.of(
                            "dtype", "F32", "rank", "2", "rhs_dtype", "F16", "rhs_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.matmul", Map.of(
                            "dtype", "F32", "rank", "1", "rhs_dtype", "F32", "rhs_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.matmul", Map.of(
                            "dtype", "F32", "rank", "2", "rhs_dtype", "F32", "rhs_rank", "3")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.matmul", Map.of(
                            "dtype", "F32", "rank", "2", "rhs_dtype", "F32")).support());

            for (String operation : Set.of("tensor.add", "tensor.sub", "tensor.mul", "tensor.div")) {
                assertEquals(OperationSupport.UNKNOWN,
                        session.operationSupport(operation, Map.of()).support());
                assertEquals(OperationSupport.UNKNOWN,
                        session.operationSupport(operation, Map.of("dtype", "F32")).support());
                assertEquals(OperationSupport.UNKNOWN,
                        session.operationSupport(operation, Map.of("rhs_dtype", "F32")).support());
                var bothF32 = session.operationSupport(operation,
                        Map.of("dtype", "F32", "rhs_dtype", "F32"));
                assertEquals(OperationSupport.DELEGATED, bothF32.support());
                assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID, bothF32.effectiveBackendId());
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation,
                                Map.of("dtype", "F16", "rhs_dtype", "F32")).support());
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation,
                                Map.of("dtype", "F32", "rhs_dtype", "F16")).support());
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation,
                                Map.of("dtype", "Q4_0", "rhs_dtype", "Q4_0")).support());
            }
            for (String operation : Set.of("tensor.add_scalar", "tensor.mul_scalar", "tensor.div_scalar")) {
                assertEquals(OperationSupport.UNKNOWN,
                        session.operationSupport(operation, Map.of()).support());
                assertEquals(OperationSupport.DELEGATED,
                        session.operationSupport(operation, Map.of("dtype", "F32")).support());
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation, Map.of("dtype", "F16")).support());
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation, Map.of("dtype", "Q4_0")).support());
            }
            assertInstanceOf(CpuBackend.class,
                    session.computeBackends().get(HatComputeProvider.CPU_FALLBACK_BACKEND_ID));
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
                    session.operationSupport("tensor.mean", Map.of()).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.sum", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "rank", "2", "dim", "-1", "keep_dim", "true")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.sum", Map.of(
                            "dtype", "F32", "rank", "2", "dim", "-2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "rank", "2", "dim", "2")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "dim", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "rank", "0", "dim", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.max", Map.of(
                            "dtype", "F32", "rank", "2", "dim", "0")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "rank", "2", "dim", "1",
                            "dim_size", "4", "numel", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "rank", "2", "dim", "1", "dim_size", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.mean", Map.of(
                            "dtype", "F32", "numel", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.max", Map.of(
                            "dtype", "F32", "numel", "0")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.sum", Map.of(
                            "dtype", "F32", "numel", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.sum", Map.of("dtype", "F16")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.softmax", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.softmax", Map.of("dtype", "F32", "rank", "3")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.softmax", Map.of(
                            "dtype", "F32", "rank", "3", "dim", "1")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.softmax", Map.of(
                            "dtype", "F32", "rank", "3", "dim", "-1")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.softmax", Map.of(
                            "dtype", "F32", "rank", "3", "dim", "3")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.log_softmax", Map.of(
                            "dtype", "F32", "rank", "3")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.log_softmax", Map.of(
                            "dtype", "F32", "rank", "3", "dim", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.log_softmax", Map.of(
                            "dtype", "F32", "rank", "3", "dim", "-1")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.softmax", Map.of(
                            "dtype", "F16", "rank", "3")).support());
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
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.attention", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.attention", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "2", "key_rank", "2", "value_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention", Map.of(
                            "dtype", "F32", "key_dtype", "F16", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "2", "value_rank", "4")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention.masked", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention.masked", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.upsample_nearest2d", Map.of()).support());
            Map<String, String> validUpsample = Map.of(
                    "dtype", "F32", "rank", "4",
                    "input_height", "2", "input_width", "2",
                    "output_height", "4", "output_width", "4");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.upsample_nearest2d", validUpsample).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.upsample_nearest2d", validUpsample).effectiveBackendId());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.upsample_nearest2d", Map.of(
                            "dtype", "F16", "rank", "4",
                            "input_height", "2", "input_width", "2",
                            "output_height", "4", "output_width", "4")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.upsample_nearest2d", Map.of(
                            "dtype", "F32", "rank", "3",
                            "input_height", "2", "input_width", "2",
                            "output_height", "4", "output_width", "4")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.upsample_nearest2d", Map.of(
                            "dtype", "F32", "rank", "4",
                            "input_height", "0", "input_width", "2",
                            "output_height", "4", "output_width", "4")).support());
            Map<String, String> validGqa = Map.of(
                    "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                    "rank", "4", "key_rank", "4", "value_rank", "4",
                    "query_heads", "8", "key_value_heads", "2", "causal", "true");
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention.gqa", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention.gqa", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "query_heads", "8", "causal", "true")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.attention.gqa", validGqa).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.attention.gqa", validGqa).effectiveBackendId());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.gqa", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "query_heads", "7", "key_value_heads", "2", "causal", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.gqa", Map.of(
                            "dtype", "F32", "key_dtype", "F16", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "query_heads", "8", "key_value_heads", "2", "causal", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.gqa", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "2", "key_rank", "4", "value_rank", "4",
                            "query_heads", "8", "key_value_heads", "2", "causal", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.gqa", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "query_heads", "8", "key_value_heads", "2", "causal", "false",
                            "mask_dtype", "F32", "mask_rank", "3")).support());
            Map<String, String> maskedAttention = Map.of(
                    "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                    "rank", "4", "key_rank", "4", "value_rank", "4",
                    "mask_dtype", "F32", "mask_rank", "2", "causal", "true");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.attention.masked", maskedAttention).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.attention.masked", maskedAttention).effectiveBackendId());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.attention.masked", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4", "causal", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.masked", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "2", "key_rank", "2", "value_rank", "2", "causal", "true")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.masked", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "mask_dtype", "F16", "mask_rank", "2", "causal", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.attention.masked", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "mask_dtype", "F32", "mask_rank", "3", "causal", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.attention.masked", Map.of(
                            "dtype", "F32", "key_dtype", "F32", "value_dtype", "F32",
                            "rank", "4", "key_rank", "4", "value_rank", "4",
                            "mask_dtype", "F32", "causal", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.rope", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.rope", Map.of("dtype", "F32", "rank", "4")).support());
            Map<String, String> validRope = Map.of(
                    "dtype", "F32", "rank", "4", "batch", "1", "heads", "8",
                    "sequence", "16", "head_dim", "64", "position", "0", "theta", "10000");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.rope", validRope).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.rope", validRope).effectiveBackendId());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rope", Map.of(
                            "dtype", "F16", "rank", "4", "batch", "1", "heads", "8",
                            "sequence", "16", "head_dim", "64", "position", "0", "theta", "10000")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rope", Map.of(
                            "dtype", "F32", "rank", "3", "batch", "1", "heads", "8",
                            "sequence", "16", "head_dim", "64", "position", "0", "theta", "10000")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rope", Map.of(
                            "dtype", "F32", "rank", "4", "batch", "1", "heads", "8",
                            "sequence", "16", "head_dim", "63", "position", "0", "theta", "10000")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rope", Map.of(
                            "dtype", "F32", "rank", "4", "batch", "1", "heads", "8",
                            "sequence", "16", "head_dim", "64", "position", "-1", "theta", "10000")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rope", Map.of(
                            "dtype", "F32", "rank", "4", "batch", "1", "heads", "8",
                            "sequence", "16", "head_dim", "64", "position", "0", "theta", "1")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.unknown", Map.of()).support());
        } finally {
            provider.close();
        }
        assertThrows(IllegalStateException.class, session::computeBackends);
    }

    @Test
    void unmaskedAttentionStagesHatOwnedHostTensorsIntoCpuFallback() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        try (var query = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 0f}, hat, 1, 2);
                var key = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 0f}, hat, 1, 2);
                var value = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {3f, 4f}, hat, 1, 2);
                var output = hat.attention(query, key, value)) {
            assertEquals(cpu, output.backend());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new float[] {3f, 4f}, output.toFloatArray(), 1.0e-5f);
        } finally {
            hat.close();
            cpu.close();
        }
    }

    @Test
    void maskedAttentionStagesQueryKeyValueAndAdditiveMaskIntoCpuFallback() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        float negativeInfinity = Float.NEGATIVE_INFINITY;
        try (var query = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {0f, 0f}, hat, 1, 1, 2, 1);
                var key = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {0f, 0f}, hat, 1, 1, 2, 1);
                var value = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {3f, 7f}, hat, 1, 1, 2, 1);
                var mask = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {0f, negativeInfinity, 0f, 0f}, hat, 2, 2);
                var output = hat.attention(query, key, value, mask, true)) {
            assertEquals(cpu, output.backend());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new float[] {3f, 5f}, output.toFloatArray(), 1.0e-5f);
        } finally {
            hat.close();
            cpu.close();
        }
    }

    @Test
    void groupedQueryAttentionStagesDifferentQueryAndKeyValueHeadCountsIntoCpuFallback() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        try (var query = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 0f, 0f, 1f}, hat, 1, 2, 1, 2);
                var key = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 0f}, hat, 1, 1, 1, 2);
                var value = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {3f, 4f}, hat, 1, 1, 1, 2);
                var output = hat.attentionGqa(query, key, value, null, false)) {
            assertEquals(cpu, output.backend());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new float[] {3f, 4f, 3f, 4f}, output.toFloatArray(), 1.0e-5f);
        } finally {
            hat.close();
            cpu.close();
        }
    }

    @Test
    void nearestNeighborUpsamplingStagesHatOwnedInputIntoCpuFallback() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        try (var input = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 2f, 3f, 4f}, hat, 1, 1, 2, 2);
                var output = hat.upsampleNearest2d(input, 4, 4)) {
            assertEquals(cpu, output.backend());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new float[] {
                        1f, 1f, 2f, 2f,
                        1f, 1f, 2f, 2f,
                        3f, 3f, 4f, 4f,
                        3f, 3f, 4f, 4f
                    }, output.toFloatArray(), 1.0e-5f);
        } finally {
            hat.close();
            cpu.close();
        }
    }

    @Test
    void catRequiresCompleteCompatibleInputMetadata() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.cat", Map.of()).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.cat", Map.of(
                            "input_dtypes", "F32,F32", "input_shapes", "2,3;2,4")).support());
            Map<String, String> valid = Map.of(
                    "input_dtypes", "F32,F32", "input_shapes", "2,3;2,4", "axis", "1");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.cat", valid).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.cat", valid).effectiveBackendId());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.cat", Map.of(
                            "input_dtypes", "F32,F16", "input_shapes", "2,3;2,4", "axis", "1")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.cat", Map.of(
                            "input_dtypes", "F32,F32", "input_shapes", "2,3;3,4", "axis", "1")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.cat", Map.of(
                            "input_dtypes", "F32,F32", "input_shapes", "2,3;2,4", "axis", "2")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void catStagesEveryHatOwnedInputIntoCpuFallback() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        try (var first = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 2f}, hat, 1, 2);
                var second = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {3f, 4f, 5f}, hat, 1, 3);
                var output = hat.cat(java.util.List.of(first, second), 1)) {
            assertEquals(cpu, output.backend());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new float[] {1f, 2f, 3f, 4f, 5f}, output.toFloatArray(), 1.0e-5f);
        } finally {
            hat.close();
            cpu.close();
        }
    }

    @Test
    void signFallbackExecutesOnCpuAndReturnsSignedValues() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        try (var input = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {-4f, 0f, 3f}, hat, 3);
                var output = hat.sign(input)) {
            assertEquals(cpu, output.backend());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new float[] {-1f, 0f, 1f}, output.toFloatArray(), 1.0e-6f);
        } finally {
            hat.close();
            cpu.close();
        }
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

    @Test
    void embeddingRequiresIntegralIndicesAndF32RankTwoWeights() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.embedding", Map.of("dtype", "I32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.embedding", Map.of(
                            "dtype", "I32", "weight_dtype", "F32", "weight_rank", "2")).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.embedding", Map.of(
                            "dtype", "I8", "weight_dtype", "F32", "weight_rank", "2")).effectiveBackendId());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.embedding", Map.of(
                            "dtype", "F32", "weight_dtype", "F32", "weight_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.embedding", Map.of(
                            "dtype", "I32", "weight_dtype", "F16", "weight_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.embedding", Map.of(
                            "dtype", "I32", "weight_dtype", "F32", "weight_rank", "1")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.embedding", Map.of(
                            "dtype", "I32", "weight_dtype", "F32", "weight_rank", "unknown")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void powerRequiresF32ForCpuFallback() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.pow", Map.of()).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.pow", Map.of("dtype", "F32")).support());
            assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                    session.operationSupport("tensor.pow", Map.of("dtype", "F32")).effectiveBackendId());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.pow", Map.of("dtype", "F16")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void sliceRequiresF32AndRankMatchedOffsetsAndSizes() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.slice", Map.of("dtype", "F32", "rank", "2")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.slice", Map.of(
                            "dtype", "F32", "rank", "2", "offsets_rank", "2", "sizes_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.slice", Map.of(
                            "dtype", "F32", "rank", "2", "offsets_rank", "1", "sizes_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.slice", Map.of(
                            "dtype", "F16", "rank", "2", "offsets_rank", "2", "sizes_rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.slice", Map.of(
                            "dtype", "F32", "rank", "0", "offsets_rank", "0", "sizes_rank", "0")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void transposeValidatesDtypeRankAndExplicitDimensions() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.transpose", Map.of("dtype", "F32")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.transpose", Map.of("dtype", "F32", "rank", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.transpose", Map.of("dtype", "F32", "rank", "1")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.transpose", Map.of(
                            "dtype", "F32", "rank", "3", "dim0", "0", "dim1", "2")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.transpose", Map.of(
                            "dtype", "F32", "rank", "3", "dim0", "0", "dim1", "3")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.transpose", Map.of(
                            "dtype", "F32", "rank", "3", "dim0", "0")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.transpose", Map.of("dtype", "Q4_0", "rank", "2")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void splitValidatesAxisAndRequiresEvenPositivePartitioning() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.split", Map.of("dtype", "F32", "rank", "2")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.split", Map.of(
                            "dtype", "F32", "rank", "2", "axis", "1", "parts", "2", "axis_size", "8")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.split", Map.of(
                            "dtype", "F32", "rank", "2", "axis", "2", "parts", "2", "axis_size", "8")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.split", Map.of(
                            "dtype", "F32", "rank", "2", "axis", "1", "parts", "3", "axis_size", "8")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.split", Map.of(
                            "dtype", "F32", "rank", "2", "axis", "1", "parts", "0", "axis_size", "8")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.split", Map.of(
                            "dtype", "F16", "rank", "2", "axis", "1", "parts", "2", "axis_size", "8")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void unaryElementwiseOperationsRequireF32ForCpuFallback() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            for (String operation : Set.of(
                    "tensor.sign", "tensor.abs", "tensor.relu", "tensor.gelu", "tensor.silu",
                    "tensor.sigmoid", "tensor.tanh", "tensor.sqrt", "tensor.exp", "tensor.log")) {
                assertEquals(OperationSupport.UNKNOWN,
                        session.operationSupport(operation, Map.of()).support(), operation);
                assertEquals(OperationSupport.DELEGATED,
                        session.operationSupport(operation, Map.of("dtype", "F32")).support(), operation);
                assertEquals(HatComputeProvider.CPU_FALLBACK_BACKEND_ID,
                        session.operationSupport(operation, Map.of("dtype", "F32")).effectiveBackendId(),
                        operation);
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation, Map.of("dtype", "F16")).support(), operation);
                assertEquals(OperationSupport.UNSUPPORTED,
                        session.operationSupport(operation, Map.of("dtype", "Q4_0")).support(), operation);
            }
        } finally {
            provider.close();
        }
    }
    @Test
    void castAndDeviceIdentityPreserveHatOwnershipWithoutUnsafeCpuFallback() {
        CpuBackend cpu = new CpuBackend();
        HatComputeBackend hat = new HatComputeBackend(cpu);
        try (var input = tech.kayys.aljabr.core.tensor.TensorFactory.of(
                    new float[] {1f, 2f}, hat, 2)) {
            assertSame(input, hat.cast(input, tech.kayys.aljabr.core.tensor.DType.F32));
            assertSame(input, hat.to(input, tech.kayys.aljabr.core.tensor.DeviceType.CPU));
            assertThrows(UnsupportedOperationException.class,
                    () -> hat.cast(input, tech.kayys.aljabr.core.tensor.DType.Q8_0));
        } finally {
            hat.close();
            cpu.close();
        }
    }

}
