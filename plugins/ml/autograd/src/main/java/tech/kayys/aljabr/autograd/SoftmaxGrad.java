package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;

/**
 * SOFTMAX BACKWARD
 * Formula:
 * dX = Y * (dY - sum(dY * Y))
 * @author bhangun
 */
public final class SoftmaxGrad implements GradFn {
        @Override
        public Map<GValueId, GValueId> backward(
                        GOp op,
                        GValueId gradOut,
                        GradContext ctx) {
                GValueRef x = op.inputs().get(0);
                GValueId y = new GValueId(op.name() + "_y"); // forward output
                GValueId dot = new GValueId(op.name() + "_dot");
                GValueId sub = new GValueId(op.name() + "_sub");
                GValueId dX = new GValueId(op.name() + "_dx");
                // dot = sum(gradOut * y)
                ctx.addOp(new GOp(
                                "mul", op.name() + "_mul",
                                List.of(new GValueRef(gradOut), new GValueRef(y)),
                                List.of(dot),
                                Map.of()));
                // sub = gradOut - dot
                ctx.addOp(new GOp(
                                "sub",
                                op.name() + "_sub",
                                List.of(new GValueRef(gradOut), new GValueRef(dot)),
                                List.of(sub),
                                Map.of()));
                // dX = y * sub
                ctx.addOp(new GOp(
                                "mul",
                                op.name() + "_final",
                                List.of(new GValueRef(y), new GValueRef(sub)),
                                List.of(dX),
                                Map.of()));
                return Map.of(x.id(), dX);
        }
}