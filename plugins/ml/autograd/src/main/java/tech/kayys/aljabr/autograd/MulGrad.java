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
public final class MulGrad implements GradFn {
        @Override
        public Map<GValueId, GValueId> backward(
                        GOp op,
                        GValueId gradOut,
                        GradContext ctx) {
                GValueRef a = op.inputs().get(0);
                GValueRef b = op.inputs().get(1);
                GValueId da = new GValueId(op.name() + "_da");
                GValueId db = new GValueId(op.name() + "_db");

                // da = gradOut * b
                ctx.addOp(new GOp("mul", op.name() + "_gradA",
                                List.of(new GValueRef(gradOut), b), List.of(da), Map.of()));
                // db = gradOut * a
                ctx.addOp(new GOp("mul", op.name() + "_gradB",
                                List.of(new GValueRef(gradOut), a), List.of(db), Map.of()));

                return Map.of(a.id(), da, b.id(), db);
        }
}
