package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;

/**
 * LAYERNORM BACKWARD
 * Key formula (simplified)
 * dX = (1/N) * gamma * (var + eps)^(-1/2) *
 * (N*dY - sum(dY) - x
 * ̂ * sum(dY * x
 * ̂))
 * @author bhangun
 */
public final class LayerNormGrad implements GradFn {
        @Override
        public Map<GValueId, GValueId> backward(
                        GOp op,
                        GValueId gradOut,
                        GradContext ctx) {
                GValueRef x = op.inputs().get(0);
                GValueRef gamma = op.inputs().get(1);
                GValueId dx = new GValueId(op.name() + "_dx");
                GValueId dgamma = new GValueId(op.name() + "_dgamma");
                GValueId dbeta = new GValueId(op.name() + "_dbeta");
                // simplified IR-level ops
                ctx.addOp(new GOp(
                                "layernorm_backward",
                                op.name() + "_backward",
                                List.of(
                                                x,
                                                gamma,
                                                new GValueRef(gradOut)),
                                List.of(dx, dgamma, dbeta),
                                Map.of()));
                return Map.of(
                                x.id(), dx,
                                gamma.id(), dgamma);
        }
}