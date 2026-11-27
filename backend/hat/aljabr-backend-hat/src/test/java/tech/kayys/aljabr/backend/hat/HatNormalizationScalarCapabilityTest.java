package tech.kayys.aljabr.backend.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.ProviderContext;

class HatNormalizationScalarCapabilityTest {
    @Test
    void normalizationRejectsInvalidEpsilonButPreservesOmittedDefaults() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            Map<String, String> base = Map.of(
                    "dtype", "F32", "weight_present", "false", "bias_present", "false");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.layer_norm", base).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "eps", "1.0E-5",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "eps", "-0.1",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "eps", "NaN",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "eps", "not-a-number",
                            "weight_present", "false", "bias_present", "false")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void batchNormValidatesMomentumWhenProvided() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            Map<String, String> base = Map.of(
                    "dtype", "F32", "training", "false",
                    "running_mean_dtype", "F32", "running_var_dtype", "F32",
                    "rank", "4", "weight_present", "false", "bias_present", "false");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", base).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "momentum", "0.1",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "4", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "momentum", "0",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "4", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "momentum", "1.1",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "4", "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "momentum", "invalid",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "rank", "4", "weight_present", "false", "bias_present", "false")).support());
        } finally {
            provider.close();
        }
    }
}
