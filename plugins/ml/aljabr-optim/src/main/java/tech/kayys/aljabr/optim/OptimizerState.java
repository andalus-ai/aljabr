package tech.kayys.aljabr.optim;

import java.util.HashMap;
import java.util.Map;
import tech.kayys.aljabr.core.tensor.Tensor;

/** Internal tensor state shared by stateful optimizer implementations. */
final class OptimizerState {
    final Map<String, Tensor> firstMoment = new HashMap<>();
    final Map<String, Tensor> secondMoment = new HashMap<>();
    final Map<String, Tensor> parameterSnapshots = new HashMap<>();
    final Map<String, Long> parameterSteps = new HashMap<>();

    void resetParameter(String name) {
        Tensor first = firstMoment.remove(name);
        if (first != null) {
            first.release();
        }
        Tensor second = secondMoment.remove(name);
        if (second != null) {
            second.release();
        }
        parameterSnapshots.remove(name);
        parameterSteps.remove(name);
    }

    void clear() {
        firstMoment.keySet().stream().toList().forEach(this::resetParameter);
        secondMoment.keySet().stream().toList().forEach(this::resetParameter);
        parameterSnapshots.keySet().stream().toList().forEach(this::resetParameter);
        parameterSteps.clear();
    }
}
