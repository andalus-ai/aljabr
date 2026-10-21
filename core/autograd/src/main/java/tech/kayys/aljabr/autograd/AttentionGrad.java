package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;

/**
 * Forward recap:
 * Q, K, V
 * A = softmax(QKᵀ / √d)
 * O = A V
 * Backward idea:
 * dV = Aᵀ dO
 * dA = dO Vᵀ
 * dQ, dK from softmax + matmul chain
 */

// Ref:
// https://arxiv.org/abs/2005.08314
// "Training Deep Nets with Subquadratic Superposition"/**
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
public final class AttentionGrad implements GradFn {
        @Override
        public Map<GValueId, GValueId> backward(
                        GOp op,
                        GValueId gradOut,
                        GradContext ctx) {
                GValueRef Q = op.inputs().get(0);
                GValueRef K = op.inputs().get(1);
                GValueRef V = op.inputs().get(2);
                GValueId dQ = new GValueId(op.name() + "_dQ");
                GValueId dK = new GValueId(op.name() + "_dK");
                GValueId dV = new GValueId(op.name() + "_dV");
                // use fused kernel instead of exploding graph
                ctx.addOp(new GOp(
                                "attention_backward",
                                op.name() + "_backward",
                                List.of(
                                                Q, K, V,
                                                new GValueRef(gradOut)),
                                List.of(dQ, dK, dV), Map.of()));
                return Map.of(
                                Q.id(), dQ,
                                K.id(), dK,
                                V.id(), dV);
        }
}
