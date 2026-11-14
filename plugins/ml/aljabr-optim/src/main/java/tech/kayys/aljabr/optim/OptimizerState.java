package tech.kayys.aljabr.optim;

import java.util.HashMap;
import java.util.Map;
import tech.kayys.aljabr.core.tensor.Tensor;

/** Internal tensor state shared by stateful optimizer implementations. */
final class OptimizerState {
    final Map<String, Tensor> firstMoment = new HashMap<>();
    final Map<String, Tensor> secondMoment = new HashMap<>();
    final Map<String, Tensor> parameterSnapshots = new HashMap<>();
    long step;

    void clear() {
        firstMoment.values().forEach(Tensor::release);
        secondMoment.values().forEach(Tensor::release);
        firstMoment.clear();
        secondMoment.clear();
        parameterSnapshots.clear();
        step = 0;
    }
}
