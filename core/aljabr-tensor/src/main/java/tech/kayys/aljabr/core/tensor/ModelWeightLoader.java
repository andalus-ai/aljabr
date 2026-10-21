package tech.kayys.aljabr.core.tensor;

import java.nio.file.Path;

/**
 * Loader for model weights from files.
 *
 * @author bhangun
 * @since 0.1.0
 */
public interface ModelWeightLoader {
    boolean supports(Path path);

    WeightAdapter load(Path path);
}
