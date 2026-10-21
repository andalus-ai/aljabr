package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;

/**
 * GELU Backward
 * dX = dY * (0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * + 0.5 * x * sech²(...) * √(2/π) * (1 + 3*0.044715*x²))
 * @author bhangun
 */
public final class GeluGrad implements GradFn {
    @Override
    public Map<GValueId, GValueId> backward(
            GOp op,
            GValueId gradOut,
            GradContext ctx) {
        GValueRef x = op.inputs().get(0);
        GValueId dX = new GValueId(op.name() + "_dx");

        // Use fused kernel or composite ops
        ctx.addOp(new GOp(
                "gelu_backward",
                op.name() + "_backward",
                List.of(x, new GValueRef(gradOut)),
                List.of(dX),
                Map.of()));

        return Map.of(x.id(), dX);
    }
}