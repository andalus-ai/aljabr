package tech.kayys.aljabr.core.tensor.lazy;

import java.util.List;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.autograd.BackwardFunction;

/**
 * Supplies local vector-Jacobian products for arithmetic graph operations.
 *
 * <p>Rules are intentionally limited to operations whose current Tensor API
 * expresses the derivative without introducing backend-specific behavior or
 * assumptions about broadcasting. Unsupported operations remain unregistered.</p>
 */
final class OperationBackwardFunctions {

    private OperationBackwardFunctions() {
    }

    static BackwardFunction forOperation(OpType operation) {
        return switch (operation) {
            case ADD -> OperationBackwardFunctions::add;
            case SUB -> OperationBackwardFunctions::subtract;
            case MUL -> OperationBackwardFunctions::multiply;
            case DIV -> OperationBackwardFunctions::divide;
            case ADD_SCALAR -> OperationBackwardFunctions::addScalar;
            case MUL_SCALAR -> OperationBackwardFunctions::multiplyScalar;
            case DIV_SCALAR -> OperationBackwardFunctions::divideScalar;
            default -> null;
        };
    }

    private static Tensor[] add(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient, gradient };
    }

    private static Tensor[] subtract(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient, gradient.mul(-1.0f) };
    }

    private static Tensor[] multiply(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] {
                gradient.mul(inputs.get(1)),
                gradient.mul(inputs.get(0))
        };
    }

    private static Tensor[] divide(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor numerator = inputs.get(0);
        Tensor denominator = inputs.get(1);
        Tensor denominatorSquared = denominator.mul(denominator);
        return new Tensor[] {
                gradient.div(denominator),
                gradient.mul(numerator).div(denominatorSquared).mul(-1.0f)
        };
    }

    private static Tensor[] addScalar(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient };
    }

    private static Tensor[] multiplyScalar(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.mul((float) arguments[0]) };
    }

    private static Tensor[] divideScalar(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.div((float) arguments[0]) };
    }
}
