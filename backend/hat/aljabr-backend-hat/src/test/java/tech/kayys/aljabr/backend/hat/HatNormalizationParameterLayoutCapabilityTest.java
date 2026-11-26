package tech.kayys.aljabr.backend.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.OperationSupport;
import tech.kayys.aljabr.spi.provider.ProviderContext;

class HatNormalizationParameterLayoutCapabilityTest {
    @Test
    void layerNormValidatesSuppliedAffineShapeAndLayoutMetadata() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            Map<String, String> valid = Map.of(
                    "dtype", "F32", "weight_present", "true", "weight_dtype", "F32",
                    "bias_present", "true", "bias_dtype", "F32",
                    "normalized_numel", "8", "weight_numel", "8", "bias_numel", "8",
                    "weight_contiguous", "true", "bias_contiguous", "true",
                    "weight_storage_offset", "0", "bias_storage_offset", "0");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.layer_norm", valid).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.layer_norm", with(valid, "weight_numel", "7")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.layer_norm", with(valid, "bias_contiguous", "false")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.layer_norm", with(valid, "weight_storage_offset", "1")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.layer_norm", with(valid, "weight_numel", "unknown")).support());

            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.layer_norm", Map.of(
                            "dtype", "F32", "weight_present", "false", "bias_present", "false")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void rmsNormValidatesWeightAgainstFinalDimensionWhenMetadataIsAvailable() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            Map<String, String> valid = Map.of(
                    "dtype", "F32", "weight_dtype", "F32",
                    "last_dim", "16", "weight_numel", "16",
                    "weight_contiguous", "true", "weight_storage_offset", "0");
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.rms_norm", valid).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rms_norm", with(valid, "weight_numel", "8")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.rms_norm", with(valid, "weight_contiguous", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.rms_norm", with(valid, "last_dim", "bad")).support());
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.rms_norm", Map.of(
                            "dtype", "F32", "weight_dtype", "F32")).support());
        } finally {
            provider.close();
        }
    }

    @Test
    void batchNormValidatesAffineAndRunningStatisticVectorLengths() throws Exception {
        HatComputeProvider provider = new HatComputeProvider();
        provider.initialize(new ProviderContext("test", Map.of()));
        try (ProviderSession session = provider.openSession(provider.devices().get(0))) {
            Map<String, String> valid = Map.ofEntries(
                    Map.entry("dtype", "F32"), Map.entry("training", "false"), Map.entry("rank", "4"),
                    Map.entry("running_mean_dtype", "F32"), Map.entry("running_var_dtype", "F32"),
                    Map.entry("weight_present", "true"), Map.entry("weight_dtype", "F32"),
                    Map.entry("bias_present", "true"), Map.entry("bias_dtype", "F32"),
                    Map.entry("channels", "32"), Map.entry("weight_numel", "32"), Map.entry("bias_numel", "32"),
                    Map.entry("running_mean_numel", "32"), Map.entry("running_var_numel", "32"),
                    Map.entry("weight_contiguous", "true"), Map.entry("bias_contiguous", "true"),
                    Map.entry("running_mean_contiguous", "true"), Map.entry("running_var_contiguous", "true"));
            assertEquals(OperationSupport.DELEGATED,
                    session.operationSupport("tensor.batch_norm", valid).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", with(valid, "running_var_numel", "31")).support());
            assertEquals(OperationSupport.UNSUPPORTED,
                    session.operationSupport("tensor.batch_norm", with(valid, "bias_contiguous", "false")).support());
            assertEquals(OperationSupport.UNKNOWN,
                    session.operationSupport("tensor.batch_norm", with(valid, "channels", "bad")).support());
        } finally {
            provider.close();
        }
    }

    private static Map<String, String> with(Map<String, String> source, String key, String value) {
        java.util.HashMap<String, String> copy = new java.util.HashMap<>(source);
        copy.put(key, value);
        return Map.copyOf(copy);
    }
}
