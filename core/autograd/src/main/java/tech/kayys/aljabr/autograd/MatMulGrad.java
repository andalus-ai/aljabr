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
public final class MatMulGrad implements GradFn {
        @Override
        public Map<GValueId, GValueId> backward(
                        GOp op,
                        GValueId gradOut,
                        GradContext ctx) {
                GValueRef A = op.inputs().get(0);
                GValueRef B = op.inputs().get(1);
                GValueId dA = new GValueId(op.name() + "_dA");
                GValueId dB = new GValueId(op.name() + "_dB");
                // dA = gradOut @ B^T
                ctx.addOp(new GOp(
                                "matmul",
                                op.name() + "_gradA",
                                List.of(
                                                new GValueRef(gradOut),
                                                transpose(B)),
                                List.of(dA),
                                Map.of()));
                // dB = A^T @ gradOut
                ctx.addOp(new GOp(
                                "matmul",
                                op.name() + "_gradB",
                                List.of(
                                                transpose(A),
                                                new GValueRef(gradOut)),
                                List.of(dB),
                                Map.of()));
                return Map.of(
                                A.id(), dA,
                                B.id(), dB);
        }

        private GValueRef transpose(GValueRef v) {
                return new GValueRef(new GValueId(v.id().id() + "_T"));
        }

}