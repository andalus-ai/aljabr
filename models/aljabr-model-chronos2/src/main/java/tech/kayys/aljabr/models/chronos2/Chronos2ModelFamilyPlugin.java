package tech.kayys.aljabr.models.chronos2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

/**
 * Model-family identity and checkpoint metadata for Amazon/AutoGluon Chronos-2.
 *
 * <p>The plugin intentionally does not claim direct Safetensors inference
 * until the Chronos-2 patch encoder, time/group attention, and quantile head
 * are implemented and validated against the reference implementation.
 */
@ApplicationScoped
public final class Chronos2ModelFamilyPlugin implements ModelFamilyPlugin {

    public static final String FAMILY_ID = "chronos2";

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                FAMILY_ID,
                "Chronos-2",
                List.of("chronos2"),
                List.of("Chronos2Model"),
                List.of(ModelFamilyCapability.TIME_SERIES_FORECASTING),
                Map.ofEntries(
                        Map.entry("bundle_profile", "optional"),
                        Map.entry("model_type", "t5"),
                        Map.entry("checkpoint_format", "safetensors"),
                        Map.entry("checkpoint_repository", "autogluon/chronos-2"),
                        Map.entry("upstream_reference_repository", "amazon/chronos-2"),
                        Map.entry("task", "zero_shot_time_series_forecasting"),
                        Map.entry("forecasting_modes", "univariate,multivariate,covariate_informed"),
                        Map.entry("covariates", "past_real,past_categorical,future_real,future_categorical"),
                        Map.entry("output", "multi_step_quantile_forecasts"),
                        Map.entry("default_context_length", "8192"),
                        Map.entry("maximum_prediction_length", "1024"),
                        Map.entry("parameter_count", "120000000"),
                        Map.entry("default_dtype", "float32"),
                        Map.entry("direct_safetensor", "pending_encoder_execution_and_tensor_binding"),
                        Map.entry("runner_status", "input_assembly_and_checkpoint_layout_validation"),
                        Map.entry("version", "0.1.0-SNAPSHOT")));
    }
}
