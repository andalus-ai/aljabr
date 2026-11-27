package tech.kayys.aljabr.backend.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.ProviderContext;

class HatBatchNormRunningStatsCapabilityTest {
    @Test
    void batchNormAcceptsExplicitlyAbsentRunningStatistics() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            Map<String, String> withoutStats = Map.of(
                    "dtype", "F32", "training", "false", "rank", "4",
                    "running_stats_present", "false",
                    "weight_present", "false", "bias_present", "false");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", withoutStats).support());

            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "rank", "4",
                            "running_stats_present", "false", "running_mean_dtype", "F32",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "rank", "4",
                            "running_stats_present", "maybe",
                            "weight_present", "false", "bias_present", "false")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void batchNormRequiresBothRunningStatisticsWhenPresent() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "rank", "4",
                            "running_stats_present", "true",
                            "running_mean_dtype", "F32", "running_var_dtype", "F32",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "rank", "4",
                            "running_stats_present", "true",
                            "running_mean_dtype", "F32",
                            "weight_present", "false", "bias_present", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", Map.of(
                            "dtype", "F32", "training", "false", "rank", "4",
                            "running_stats_present", "true",
                            "running_mean_dtype", "F16", "running_var_dtype", "F32",
                            "weight_present", "false", "bias_present", "false")).support());
        } finally {
            provider.close();
        }
    }
}
