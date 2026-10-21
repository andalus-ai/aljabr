package tech.kayys.aljabr.plugin.kernel;

import java.util.Map;

/**
 * SPI — context passed to a kernel plugin during execution.
 */
public interface KernelContext {
    Map<String, Object> parameters();
}
