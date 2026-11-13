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
public final class OperationBackwardFunctions {

    private OperationBackwardFunctions() {
    }

    public static BackwardFunction forOperation(OpType operation) {
        return switch (operation) {
            case ADD -> OperationBackwardFunctions::add;
            case SUB -> OperationBackwardFunctions::subtract;
            case MUL -> OperationBackwardFunctions::multiply;
            case DIV -> OperationBackwardFunctions::divide;
            case ADD_SCALAR -> OperationBackwardFunctions::addScalar;
            case MUL_SCALAR -> OperationBackwardFunctions::multiplyScalar;
            case DIV_SCALAR -> OperationBackwardFunctions::divideScalar;
            case MATMUL -> OperationBackwardFunctions::matmul;
            case MEAN -> OperationBackwardFunctions::mean;
            case SUM -> OperationBackwardFunctions::sum;
            case SIGMOID -> OperationBackwardFunctions::sigmoid;
            case TANH -> OperationBackwardFunctions::tanh;
            case EXP -> OperationBackwardFunctions::exp;
            case LOG -> OperationBackwardFunctions::log;
            case POW -> OperationBackwardFunctions::pow;
            case SQRT -> OperationBackwardFunctions::sqrt;
            default -> null;
        };
    }

    private static Tensor[] matmul(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor left = inputs.get(0);
        Tensor right = inputs.get(1);
        if (left.shape().rank() != 2 || right.shape().rank() != 2 || gradient.shape().rank() != 2) {
            throw new UnsupportedOperationException(
                    "MATMUL autograd currently supports rank-2 matrices only");
        }
        Tensor rightTranspose = right.transpose();
        Tensor leftTranspose = left.transpose();
        Tensor leftGradient = null;
        try {
            leftGradient = gradient.matmul(rightTranspose);
            Tensor rightGradient = leftTranspose.matmul(gradient);
            return new Tensor[] { leftGradient, rightGradient };
        } catch (RuntimeException | Error failure) {
            if (leftGradient != null) {
                leftGradient.release();
            }
            throw failure;
        } finally {
            rightTranspose.release();
            leftTranspose.release();
        }
    }

    private static Tensor[] mean(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor input = inputs.get(0);
        Tensor expanded = expandReductionGradient(gradient, input, arguments);
        long reductionSize = arguments.length == 0
                ? input.numel()
                : input.shape().dim(normalizeDimension((int) arguments[0], input.shape().rank()));
        try {
            return new Tensor[] { expanded.div((float) reductionSize) };
        } finally {
            expanded.release();
        }
    }

    private static Tensor[] sum(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { expandReductionGradient(gradient, inputs.get(0), arguments) };
    }

    private static Tensor expandReductionGradient(Tensor gradient, Tensor input, Object[] arguments) {
        Tensor expanded = gradient;
        boolean ownsExpanded = false;
        try {
            if (arguments.length != 0 && !(boolean) arguments[1]) {
                expanded = gradient.unsqueeze((int) arguments[0]);
                ownsExpanded = true;
            }
            Tensor ones = input.zerosLike().add(1.0f);
            try {
                return expanded.mul(ones);
            } finally {
                ones.release();
            }
        } finally {
            if (ownsExpanded) expanded.release();
        }
    }

    private static int normalizeDimension(int dimension, int rank) {
        int normalized = dimension < 0 ? rank + dimension : dimension;
        if (normalized < 0 || normalized >= rank) {
            throw new IllegalArgumentException("Reduction dimension " + dimension
                    + " is out of range for rank " + rank);
        }
        return normalized;
    }

    private static Tensor reduceToInputShape(Tensor gradient, Tensor input) {
        long[] inputDims = input.shape().dims();
        long[] gradientDims = gradient.shape().dims();
        if (java.util.Arrays.equals(inputDims, gradientDims)) return gradient.add(0.0f);
        if (inputDims.length > gradientDims.length) {
            throw new IllegalArgumentException("Cannot reduce gradient shape " + gradient.shape()
                    + " to input shape " + input.shape());
        }
        Tensor reduced = gradient;
        boolean ownsReduced = false;
        try {
            int leading = gradientDims.length - inputDims.length;
            for (int i = 0; i < leading; i++) {
                Tensor next = reduced.sum(0, false);
                if (ownsReduced) reduced.release();
                reduced = next;
                ownsReduced = true;
            }
            long[] currentDims = reduced.shape().dims();
            for (int axis = 0; axis < inputDims.length; axis++) {
                if (inputDims[axis] == currentDims[axis]) continue;
                if (inputDims[axis] != 1 || currentDims[axis] < 1) {
                    throw new IllegalArgumentException("Gradient shape " + gradient.shape()
                            + " is not broadcast-compatible with input shape " + input.shape());
                }
                Tensor next = reduced.sum(axis, true);
                if (ownsReduced) reduced.release();
                reduced = next;
                ownsReduced = true;
                currentDims = reduced.shape().dims();
            }
            return reduced.reshape(inputDims);
        } finally {
            if (ownsReduced) reduced.release();
        }
    }

    private static Tensor[] add(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] {
                reduceToInputShape(gradient, inputs.get(0)),
                reduceToInputShape(gradient, inputs.get(1))
        };
    }

    private static Tensor[] subtract(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor negative = gradient.mul(-1.0f);
        try {
            return new Tensor[] {
                    reduceToInputShape(gradient, inputs.get(0)),
                    reduceToInputShape(negative, inputs.get(1))
            };
        } finally {
            negative.release();
        }
    }

    private static Tensor[] multiply(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor leftFull = gradient.mul(inputs.get(1));
        Tensor rightFull = null;
        try {
            rightFull = gradient.mul(inputs.get(0));
            return new Tensor[] {
                    reduceToInputShape(leftFull, inputs.get(0)),
                    reduceToInputShape(rightFull, inputs.get(1))
            };
        } finally {
            leftFull.release();
            if (rightFull != null) rightFull.release();
        }
    }

    private static Tensor[] divide(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor numerator = inputs.get(0);
        Tensor denominator = inputs.get(1);
        Tensor denominatorSquared = denominator.mul(denominator);
        Tensor numeratorFull = null;
        Tensor numeratorGradient = null;
        Tensor denominatorNumerator = null;
        Tensor denominatorQuotient = null;
        Tensor denominatorNegative = null;
        Tensor denominatorGradient = null;
        try {
            numeratorFull = gradient.div(denominator);
            numeratorGradient = reduceToInputShape(numeratorFull, numerator);
            denominatorNumerator = gradient.mul(numerator);
            denominatorQuotient = denominatorNumerator.div(denominatorSquared);
            denominatorNegative = denominatorQuotient.mul(-1.0f);
            denominatorGradient = reduceToInputShape(denominatorNegative, denominator);
            return new Tensor[] { numeratorGradient, denominatorGradient };
        } catch (RuntimeException | Error failure) {
            if (numeratorGradient != null) numeratorGradient.release();
            if (denominatorGradient != null) denominatorGradient.release();
            throw failure;
        } finally {
            denominatorSquared.release();
            if (numeratorFull != null) numeratorFull.release();
            if (denominatorNumerator != null) denominatorNumerator.release();
            if (denominatorQuotient != null) denominatorQuotient.release();
            if (denominatorNegative != null) denominatorNegative.release();
        }
    }

    private static Tensor[] addScalar(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.add(0.0f) };
    }

    private static Tensor[] multiplyScalar(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.mul((float) arguments[0]) };
    }

    private static Tensor[] divideScalar(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.div((float) arguments[0]) };
    }
    private static Tensor[] sigmoid(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor zeros = output.zerosLike();
        Tensor ones = null;
        Tensor oneMinusOutput = null;
        try {
            ones = zeros.add(1.0f);
            oneMinusOutput = ones.sub(output);
            return new Tensor[] { gradient.mul(output).mul(oneMinusOutput) };
        } finally {
            zeros.release();
            if (ones != null) {
                ones.release();
            }
            if (oneMinusOutput != null) {
                oneMinusOutput.release();
            }
        }
    }

    private static Tensor[] tanh(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor squared = output.mul(output);
        Tensor negativeSquared = null;
        Tensor zeros = null;
        Tensor oneMinusSquared = null;
        try {
            negativeSquared = squared.mul(-1.0f);
            zeros = output.zerosLike();
            Tensor ones = zeros.add(1.0f);
            try {
                oneMinusSquared = ones.add(negativeSquared);
            } finally {
                ones.release();
            }
            return new Tensor[] { gradient.mul(oneMinusSquared) };
        } finally {
            squared.release();
            if (negativeSquared != null) {
                negativeSquared.release();
            }
            if (zeros != null) {
                zeros.release();
            }
            if (oneMinusSquared != null) {
                oneMinusSquared.release();
            }
        }
    }

    private static Tensor[] exp(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.mul(output) };
    }

    private static Tensor[] log(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        return new Tensor[] { gradient.div(inputs.get(0)) };
    }

    private static Tensor[] pow(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        float exponent = (float) arguments[0];
        Tensor powered = inputs.get(0).pow(exponent - 1.0f);
        try {
            Tensor scaled = powered.mul(exponent);
            try {
                return new Tensor[] { gradient.mul(scaled) };
            } finally {
                scaled.release();
            }
        } finally {
            powered.release();
        }
    }

    private static Tensor[] sqrt(Tensor output, Tensor gradient, List<Tensor> inputs, Object[] arguments) {
        Tensor denominator = output.mul(2.0f);
        try {
            return new Tensor[] { gradient.div(denominator) };
        } finally {
            denominator.release();
        }
    }

}
