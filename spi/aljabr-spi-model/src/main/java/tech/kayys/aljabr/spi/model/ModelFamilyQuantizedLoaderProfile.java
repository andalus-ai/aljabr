package tech.kayys.aljabr.spi.model;

import java.nio.file.Path;
import java.util.List;
/**
 * Immutable record representing modelfamilyquantizedloaderprofile data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelFamilyQuantizedLoaderProfile(
        boolean mobileQatSupported,
        boolean inferredFromConfig,
        String format,
        String container,
        String loaderScope,
        List<String> problemCodes) {

    public ModelFamilyQuantizedLoaderProfile {
        format = format == null ? "" : format;
        container = container == null ? "" : container;
        loaderScope = loaderScope == null ? "" : loaderScope;
        problemCodes = problemCodes == null ? List.of() : List.copyOf(problemCodes);
    }

    public static ModelFamilyQuantizedLoaderProfile fromModelDir(Path dir) {
        try {
            Path configPath = dir.resolve("config.json");
            if (!java.nio.file.Files.exists(configPath)) {
                return null;
            }
            String configContent = java.nio.file.Files.readString(configPath);
            if (!configContent.contains("quantization_config")) {
                return null;
            }

            boolean isGemma4 = configContent.contains("\"gemma4\"");
            boolean hasQuantMethodGemma = configContent.contains("\"gemma\"");
            boolean hasVision = configContent.contains("\"vision_config\"");
            boolean hasAudio = configContent.contains("\"audio_config\"");

            boolean supportsMobileQat = isGemma4 && hasQuantMethodGemma && (hasVision || hasAudio);

            if (!supportsMobileQat) {
                // If it is not mobile qat and has no other known format, we might just
                // return generic
                if (!configContent.contains("\"future_format\"")) {
                    return null;
                }
            }

            String format = supportsMobileQat ? "mobile" : "future_format";
            String container = supportsMobileQat ? "transformers" : "future_container";
            String scope = supportsMobileQat ? "metadata_only_pending_mobile_quant_loader"
                    : "metadata_only_pending_future_quant_loader";
            List<String> codes = new java.util.ArrayList<>();
            if (supportsMobileQat) {
                codes.add(ModelFamilyProblemCodes.QAT_MOBILE_LOADER_PENDING);
            } else {
                codes.add(ModelFamilyProblemCodes.QUANTIZED_WEIGHT_LOADER_PENDING);
            }

            return new ModelFamilyQuantizedLoaderProfile(
                    supportsMobileQat, true, format, container, scope, codes);
        } catch (Exception e) {
            return null;
        }
    }
}
