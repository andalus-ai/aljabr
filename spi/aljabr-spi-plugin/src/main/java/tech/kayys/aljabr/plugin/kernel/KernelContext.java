package tech.kayys.aljabr.plugin.kernel;

import java.util.Map;

/** Context passed to a kernel plugin during execution. */
public interface KernelContext {
    Map<String, Object> parameters();
}
