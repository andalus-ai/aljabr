package tech.kayys.aljabr.spi.model;
/**
 * 
 * Core class for kayys module.
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class ModelFamilyProblemCodes {
    public static final String UNSUPPORTED_ARCHITECTURE = "unsupported_architecture";
    public static final String MISSING_TOKENIZER_DESCRIPTOR = "missing_tokenizer_descriptor";
    public static final String MISSING_ARCHITECTURE_ADAPTER = "missing_architecture_adapter";
    public static final String MULTIPLE_PRIMARY_FAMILIES = "multiple_primary_families";
    public static final String NO_SUPPORTED_FAMILIES = "no_supported_families";

    public static final String QUANTIZED_WEIGHT_LOADER_PENDING = "quantized_weight_loader_pending";
    public static final String QAT_MOBILE_LOADER_PENDING = "qat_mobile_loader_pending";
    public static final String QAT_Q4_0_LOADER_PENDING = "qat_q4_0_loader_pending";

    public static final String MISSING_MODEL_FAMILY_PLUGIN = "model_family_not_found";
    public static final String AMBIGUOUS_MODEL_FAMILY = "model_family_ambiguous";

    private ModelFamilyProblemCodes() {
    }
}
