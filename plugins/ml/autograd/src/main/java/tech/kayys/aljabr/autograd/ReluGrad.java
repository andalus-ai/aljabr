package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;
/**
 * 
 * Core class for tech module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class ReluGrad implements GradFn {
    @Override
    public Map<GValueId, GValueId> backward(
            GOp op,
            GValueId gradOut,
            GradContext ctx) {
        GValueRef x = op.inputs().get(0);
        GValueId dx = new GValueId(op.name() + "_dx");

        ctx.addOp(new GOp("relu_backward", op.name() + "_backward",
                List.of(x, new GValueRef(gradOut)), List.of(dx), Map.of()));

        return Map.of(x.id(), dx);
    }
}
