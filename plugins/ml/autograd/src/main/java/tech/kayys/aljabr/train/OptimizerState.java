package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.HashMap;
import java.util.Map;
/**
 * 
 * Core class for tech module.
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class OptimizerState {
    public final Map<String, Tensor> m = new HashMap<>(); // first moment
    public final Map<String, Tensor> v = new HashMap<>(); // second moment
    public int step = 0;
}