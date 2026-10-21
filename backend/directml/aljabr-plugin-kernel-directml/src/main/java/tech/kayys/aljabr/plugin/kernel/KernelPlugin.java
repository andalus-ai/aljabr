package tech.kayys.aljabr.plugin.kernel;

import java.util.Map;
import java.util.Set;

/**
 * SPI — lifecycle contract for hardware kernel plugins.
 */
public interface KernelPlugin {
    String id();

    String name();

    String version();

    String description();

    boolean isAvailable();

    String platform();

    Set<String> supportedArchitectures();

    Set<String> supportedVersions();

    Set<String> supportedOperations();

    <T> KernelResult<T> execute(KernelOperation operation, KernelContext context) throws KernelException;

    Map<String, Object> metadata();
}
